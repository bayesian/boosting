/* Copyright 2015,2016 Tao Xu
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "boost/make_shared.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/move/unique_ptr.hpp"
#include "Concurrency.h"
#include "Config.h"
#include "GbmFun.h"
#include "Gbm.h"
#include "LogisticFun.h"
#include "DataSet.h"
#include "Tree.h"
#include "gflags/gflags.h"
#include "folly/String.h"
#include "folly/json.h"
#include "thrift/concurrency/PosixThreadFactory.h"
#include "thrift/concurrency/Thread.h"
#include "thrift/concurrency/ThreadManager.h"

using namespace boosting;
using namespace std;

DEFINE_int32(num_examples_for_bucketing, 1024*1024*5,
             "number of data points used for data set compression");

DEFINE_int32(random_seed, 123456789, "The random seed.");

DEFINE_string(config_file, "",
              "file contains the configurations");

DEFINE_string(training_files, "",
              "comma separated list of data files for training");

DEFINE_string(testing_files, "",
              "comma separated list of data files for training");

DEFINE_string(eval_output_file, "",
	      "file contains eval output:could be stdout");

DEFINE_string(model_file, "",
              "file contains the whole model");

DEFINE_bool(eval_only, false,
            "eval only mode");

DEFINE_bool(find_optimal_num_trees, false,
            "using huge data to trim number of trees");

DEFINE_int32(num_examples_for_training, -1,
             "number of data points used for training, "
             " -1 will use all available");

const int CHUNK_SIZE = 2500;  // # of lines each data loading chunk may parse


/**
 * Utility class used to parallelize dataset loading.
 */
class DataChunk : public apache::thrift::concurrency::Runnable {

public:

  DataChunk(const Config& cfg, const DataSet& dataSet,
            CounterMonitor* monitorPtr = NULL) :
    cfg_(cfg), dataSet_(dataSet), monitorPtr_(monitorPtr) {}

  bool addLine(const string& s) {
    if (s.empty()) {
      return false;
    }
    lines_.emplace_back(s);
    return true;
  }

  void parseLines() {
    featureVectors_.reserve(lines_.size());
    targets_.reserve(lines_.size());
    boost::scoped_array<double> farr(new double[cfg_.getNumFeatures()]);
    double target;
    for (const string& line : lines_) {
      if (dataSet_.getRow(line, &target, farr)) {
        targets_.push_back(target);
        featureVectors_.emplace_back(farr.get(),
                                     farr.get() + cfg_.getNumFeatures());
      }
    }
  }

  void run() {
    parseLines();
    if (monitorPtr_ != NULL) {
      monitorPtr_->decrement();
    }
  }

  const vector<vector<double>>& getFeatureVectors() const {
    return featureVectors_;
  }

  const vector<double>& getTargets() const {
    return targets_;
  }

  size_t getLineBufferSize() const {
    return lines_.size();
  }

  size_t getSize() const {
    return featureVectors_.size();
  }

  // Does not use class member dataset, since we might want to load into
  // another dataset.
  size_t addToDataSet(DataSet* dataSet) const {
    CHECK(featureVectors_.size() == targets_.size())
      << "featureVectors_ and targets_ vectors must be the same size";
    boost::scoped_array<double> farr(new double[cfg_.getNumFeatures()]);
    size_t size = featureVectors_.size();
    for (size_t i = 0; i < size; ++i) {
      const auto fvec = featureVectors_[i];
      copy(fvec.begin(), fvec.end(), farr.get());
      if (!dataSet->addVector(farr, targets_[i])) {
        return i;
      }
    }
    return size;
  }

private:

  const Config& cfg_;
  const DataSet& dataSet_;
  CounterMonitor* monitorPtr_;  // for threading purposes
  vector<string> lines_;
  vector<vector<double>> featureVectors_;
  vector<double> targets_;

};

// Divide training data file's lines into chunks,
// and parse chunks concurrently if desired/possible
void readIntoDataChunks(istream& in,
                        vector<boost::shared_ptr<DataChunk>>* chunks,
                        size_t chunkSize, const Config& cfg,
                        const DataSet& dataSet) {
  // Read lines, placing them into chunks
  CounterMonitor monitor(0);
  boost::shared_ptr<DataChunk> curChunkPtr =
    boost::make_shared<DataChunk>(cfg, dataSet, &monitor);
  string line;
  while (getline(in, line)) {
    curChunkPtr->addLine(line);
    if (curChunkPtr->getLineBufferSize() >= chunkSize) {
      // filled up current chunk, so start another one
      chunks->push_back(curChunkPtr);
      curChunkPtr = boost::make_shared<DataChunk>(cfg, dataSet, &monitor);
    }
  }
  if (curChunkPtr->getLineBufferSize() > 0) {
    chunks->push_back(curChunkPtr);
  }

  // Parse all chunks
  if (FLAGS_num_threads > 0 && !chunks->empty()) {
    monitor.init(chunks->size());
    for (auto chunkPtr : *chunks) {
      Concurrency::threadManager->add(chunkPtr);
    }
    monitor.wait();
  } else {
    for (auto chunkPtr : *chunks) {
      chunkPtr->parseLines();
    }
  }
}

// write feature importance vector
void dumpFimps(const string& fileName, const Config& cfg, double fimps[]) {
  ofstream fs(fileName);
  for (int fid = 0; fid < cfg.getNumFeatures(); fid++) {
    fs << fid << '\t' << fimps[fid] << '\t'
       << cfg.getFeatureName(fid) << '\n';
  }
  fs.close();
}

// write Json dump of boosting model
template <class T>
void dumpModel(const string& fileName,
	       const Config& cfg,
               const vector<TreeNode<T>* >& model) {
  folly::dynamic m = folly::dynamic::object;
  folly::dynamic trees = {};

  for (const auto& t : model) {
    trees.push_back(std::move(t->toJson(cfg)));
  }

  m.insert("trees", trees);

  ofstream fs(fileName);
  fs << toPrettyJson(m);
  fs.close();
}

unique_ptr<GbmFun> getGbmFun(LossFunction loss) {
  if (loss == L2Regression) {
    return unique_ptr<GbmFun>(new LeastSquareFun());
  } else {
    return unique_ptr<GbmFun>(new LogisticFun());
  }
}

int main(int argc, char **argv) {
  stringstream ss;
  for (int i = 0; i < argc; i++) {
    ss << argv[i] << " ";
  }

  google::SetUsageMessage("Gbm Training");
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  Concurrency::initThreadManager();

  // Initialize random seed.
  srand(FLAGS_random_seed);

  LOG(INFO) << ss.str();

  Config cfg;

  LOG(INFO) << "loading config";

  CHECK(cfg.readConfig(FLAGS_config_file));
  unique_ptr<GbmFun> pfun = getGbmFun(cfg.getLossFunction());
  GbmFun& fun = *pfun;

  unique_ptr<GbmFun> pCmpFun = getGbmFun(cfg.getLossFunction());
  GbmFun& cmpFun = *pCmpFun;

  vector<TreeNode<double>*> model;
  DataSet ds(cfg, FLAGS_num_examples_for_bucketing,
             FLAGS_num_examples_for_training);

  if (!FLAGS_eval_only) {
    // Compute model from training files

    // First, load training files
    vector<folly::StringPiece> sv;
    folly::split(',', FLAGS_training_files, sv);

    time_t start, end;
    time(&start);

    for (const auto& s : sv) {
      LOG(INFO) << "loading data from:" << s;

      ifstream fs(s.str());
      vector<boost::shared_ptr<DataChunk>> dataChunks;
      readIntoDataChunks(fs, &dataChunks, CHUNK_SIZE, cfg, ds);
      for (const auto chunkPtr : dataChunks) {
        chunkPtr->addToDataSet(&ds);
      }

      time(&end);
      double timespent = difftime(end, start);
      LOG(INFO) << "read " << ds.getNumExamples() << " examples in "
                << timespent << " sec" << endl;
    }

    ds.close();

    // Second, train the models
    Gbm engine(fun, ds, cfg);
    double* fimps = new double[cfg.getNumFeatures()];
    for (int i = 0; i < cfg.getNumFeatures(); i++) {
      fimps[i] = 0.0;
    }
    engine.getModel(&model, fimps);

    // Third, write the model files
    dumpFimps(FLAGS_model_file + ".fimps", cfg, fimps);
    dumpModel(FLAGS_model_file, cfg, model);
  } else {
    // Skip training, load previously written model

    LOG(INFO) << "loading model from " << FLAGS_model_file;
    ifstream fs(FLAGS_model_file);
    stringstream buffer;
    buffer << fs.rdbuf();

    const folly::dynamic obj = folly::parseJson(buffer.str());
    const int numTrees = obj["trees"].size();
    LOG(INFO) << "num trees: " << numTrees;
    model.reserve(numTrees);
    for (int i = 0; i < numTrees; i++) {
      model.push_back(fromJson<double>(obj["trees"][i], cfg));
    }
  }

  if (FLAGS_testing_files != "") {
    ostream *os = NULL;
    ofstream ofs;
    if (FLAGS_eval_output_file != "") {
      if (FLAGS_eval_output_file == "stdout") {
	os = &cout;
      } else {
	ofs.open(FLAGS_eval_output_file);
	os = &ofs;
      }
    }

    // See how well the model performs on testing data
    double target, score;
    boost::scoped_array<double> fvec(new double[cfg.getNumFeatures()]);
    int numEvalColumns = cfg.getEvalIdx().size();
    boost::scoped_array<string> feval(new string[numEvalColumns]);

    vector<unique_ptr<GbmFun>> funs;
    for (int i = 0; i < model.size(); i++) {
      funs.push_back(getGbmFun(cfg.getLossFunction()));
    }

    vector<folly::StringPiece> tsv;
    folly::split(',', FLAGS_testing_files, tsv);
    for (const auto& s : tsv) {
      LOG(INFO) << "loading data from:" << s;
      istream *is;
      fstream fs;

      if (s.str() == "stdin") {
        is = &cin;
      } else {
        fs.open(s.str());
        is = &fs;
      }
      string line;
      vector<double> scores;
      while(getline(*is, line)) {
        ds.getRow(line, &target, fvec, &score);
        double f;
        if (FLAGS_find_optimal_num_trees) {
          f = predict_vec(model, fvec, &scores);
          for (int i = 0; i < model.size(); i++) {
            funs[i]->accumulateExampleLoss(target, scores[i]);
          }
          scores.clear();
        } else {
          f = predict(model, fvec);
        }

	if (os != NULL) {
	  ds.getEvalColumns(line, feval);
	  for (int i = 0; i < numEvalColumns; i++) {
	    (*os) << feval[i] << '\t';
	  }
	  (*os) << f << endl;
	}

	fun.accumulateExampleLoss(target, f);
        cmpFun.accumulateExampleLoss(target, score);
	if (fun.getNumExamples() % 1000 == 0) {
	  LOG(INFO) << "test loss reduction: " << fun.getReduction()
		    << " on num examples: " << fun.getNumExamples()
		    << " total loss: " << fun.getLoss()
		    << " logged score: " << score
		    << " computed score: " << f
                    << " cmp loss: " << cmpFun.getLoss()
                    << " cmp reduction: " << cmpFun.getReduction();
	}
      }
    }
    if (os != NULL) {
      os->flush();
    }

    if (FLAGS_find_optimal_num_trees) {
      cout << model.size() << '\t';
      for (int i = 0; i < model.size(); i++) {
	cout << funs[i]->getLoss() << '\t';
      }
    }

    LOG(INFO) << fun.getNumExamples() << '\t' << fun.getReduction() << '\t'
	      << fun.getLoss() << endl;

    LOG(INFO) << "test loss reduction: " << fun.getReduction()
              << ", cmp loss function: " << cmpFun.getReduction()
	      << " on num examples: " << fun.getNumExamples();

  }
}
