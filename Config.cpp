/*
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

#include "Config.h"

#include <fstream>
#include <unordered_map>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <folly/json.h>

namespace boosting {

using namespace folly;
using namespace std;

bool Config::readConfig(const std::string& fileName) {
  ifstream fs(fileName);
  stringstream buffer;

  buffer << fs.rdbuf();

  try {
    const dynamic cfg = parseJson(buffer.str());
    numTrees_ = cfg["num_trees"].asInt();
    numLeaves_ = cfg["num_leaves"].asInt();
    exampleSamplingRate_ = cfg["example_sampling_rate"].asDouble();
    featureSamplingRate_ = cfg["feature_sampling_rate"].asDouble();
    learningRate_ = cfg["learning_rate"].asDouble();

    // load dictionary: indices <--> column names
    const dynamic& columnNames = cfg["all_columns"];
    unordered_map<fbstring, int> columnIdx;
    int cidx = 0;
    for (auto it = columnNames.begin(); it != columnNames.end(); ++it) {
      auto columnName = it->asString();
      allColumns_.emplace_back(columnName.toStdString());
      CHECK(columnIdx.find(columnName) == columnIdx.end());
      columnIdx[columnName] = cidx;
      cidx++;
    }

    targetIdx_ = columnIdx[cfg["target_column"].asString()];

    auto it = cfg.find("compare_column");
    cmpIdx_ = (it != cfg.items().end())
      ? columnIdx[it->second.asString()] : -1;

    it = cfg.find("loss_function");
    if (it != cfg.items().end() && it->second.asString() == "logistic") {
      lossFunction_ = L2Logistic;
    } else {
      lossFunction_ = L2Regression;
    }

    const dynamic& trainColumns = cfg["train_columns"];
    for (auto it = trainColumns.begin(); it != trainColumns.end(); ++it) {
      featureToIndexMap_[it->asString().toStdString()] = trainIdx_.size();
      trainIdx_.push_back(columnIdx.at(it->asString()));
    }

    const dynamic& weakColumns = cfg["weak_columns"];
    for (auto it = weakColumns.begin(); it != weakColumns.end(); ++it) {
      weakIdx_.push_back(columnIdx.at(it->asString()));
    }

    const dynamic& evalColumns = cfg["eval_output_columns"];
    for (auto it = evalColumns.begin(); it != evalColumns.end(); ++it) {
      evalIdx_.push_back(columnIdx.at(it->asString()));
    }

    const dynamic& targetColumn = cfg["target_column"];
    targetIdx_ = columnIdx.at(targetColumn.asString());

    const string& delimiter = cfg["delimiter"].asString().toStdString();

    if (delimiter == "TAB") {
      delimiter_ = '\t';
    } else if (delimiter == "COMMA") {
      delimiter_ = ',';
    } else if (delimiter == "CTRL-A") {
      delimiter_ = '\001';
    } else {
      LOG(FATAL) << "invalid delimiter " << delimiter;
      return false;
    }
  } catch (const exception& ex) {
    LOG(FATAL) << "parse config failed: " << ex.what();
    return false;
  }
  return true;
}

}
