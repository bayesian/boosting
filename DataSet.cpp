#include "DataSet.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <cmath>

#include "Config.h"
#include "Tree.h"
#include <gflags/gflags.h>
#include <folly/Conv.h>
#include <folly/String.h>

namespace boosting {

using namespace std;

DataSet::DataSet(const Config& cfg, int bucketingThresh, int examplesThresh)
  : cfg_(cfg), bucketingThresh_(bucketingThresh),
    examplesThresh_(examplesThresh),
    preBucketing_(true), numExamples_(0),
    numFeatures_(cfg.getNumFeatures()),
    features_(new FeatureData[numFeatures_]) {

  for (int i = 0; i < numFeatures_; i++) {
    features_[i].fvec.reset(new vector<double>());
    features_[i].encoding = DOUBLE;
  }
}

bool DataSet::getEvalColumns(const std::string& line,
			     boost::scoped_array<std::string>& feval) const {
  vector<folly::StringPiece> sv;
  folly::split(cfg_.getDelimiter(), line, sv);
  const auto& evalColumns = cfg_.getEvalIdx();

  for (int fid = 0; fid < evalColumns.size(); fid++) {
    feval[fid] = sv[evalColumns[fid]].toString();
  }
  return true;
}

bool DataSet::getRow(const string& line, double* target, double* pos,
                     boost::scoped_array<double>& fvec,
                     double* cmpValue) const {
  try {
    vector<folly::StringPiece> sv;
    folly::split(cfg_.getDelimiter(), line, sv);

    if (sv.size() != cfg_.getColumnNames().size()) {
      LOG(ERROR) << "invalid row: unexpected number of columns" << line
                 << ", expected " << cfg_.getColumnNames().size()
                 << ", got " << sv.size();
      return false;
    }
    const auto& trainColumns = cfg_.getTrainIdx();

    for (int fid = 0; fid < trainColumns.size(); fid++) {
      fvec[fid] = atof(sv[trainColumns[fid]].toString().c_str());
    }
    *target = atof(sv[cfg_.getTargetIdx()].toString().c_str());
    if (cfg_.getPosIdx() != -1) {
      double position = atof(sv[cfg_.getPosIdx()].toString().c_str());
      *pos = log(3.0)/log(position + 3);
    } else {
      *pos = 1.0;
    }
    if (cfg_.getCompareIdx() != -1 && cmpValue != NULL) {
      *cmpValue = atof(sv[cfg_.getCompareIdx()].toString().c_str());
    }
  } catch (...) {
    LOG(ERROR) << "fail to process line: " << line;
    return false;
  }
  return true;
}

//predict without explicitly creating feature vector, since it is
//expensive to copy the long vector. used only in Gbm eval step.
double DataSet::getPrediction(TreeNode<uint16_t>* rt, int eid) const {
  const PartitionNode<uint16_t>* pnode =
    dynamic_cast<const PartitionNode<uint16_t>*>(rt);

  if (pnode != NULL) {
    const int fid = pnode->getFid();
    uint16_t fv;
    if (features_[fid].encoding == BYTE) {
      fv = (*features_[fid].bvec)[eid];
    } else {
      fv = (*features_[fid].svec)[eid];
    }

    if (fv <= pnode->getFv()) {
      return getPrediction(pnode->getLeft(), eid);
    } else {
      return getPrediction(pnode->getRight(), eid);
    }
  } else {
    const LeafNode<uint16_t>* lfnode =
      dynamic_cast<const LeafNode<uint16_t>*>(rt);
    return lfnode->getVote();
  }
}

bool DataSet::addVector(const boost::scoped_array<double>& fvec,
                        double target, double pos) {
  if (examplesThresh_ != -1 && numExamples_ > examplesThresh_) {
    return false;
  }

  for (int fid = 0; fid < numFeatures_; fid++) {
    double val = fvec[fid];
    if (preBucketing_) {
      features_[fid].fvec->push_back(val);
    } else {
      const auto& transitions = features_[fid].transitions;
      const auto& it = lower_bound(transitions.begin(),
                                   transitions.end(),
                                   val);

      if (features_[fid].encoding == EMPTY) {
        continue;
      } else if (features_[fid].encoding == BYTE) {
        (features_[fid].bvec)->push_back(
          static_cast<uint8_t>(it - transitions.begin()));
      } else if (features_[fid].encoding == SHORT) {
        (features_[fid].svec)->push_back(
          static_cast<uint16_t>(it - transitions.begin()));
      } else {
        LOG(INFO) << "invalid encoding after bucketing";
      }
    }
  }
  targets_.push_back(target);
  positions_.push_back(pos);
  numExamples_++;

  if (bucketingThresh_ != -1 && numExamples_ > bucketingThresh_
      && preBucketing_) {
    bucketize();
  }
  return true;
}

struct IdVal {
  int id;
  double val;

  IdVal(int i, double v) : id(i), val(v) {
  }
};

template<class T>
void fillValues(const vector<IdVal>& idvals,
                const vector<int>& transitions,
                vector<T>& vec) {
  int idx = 0;
  for (int i = 0; i < transitions.size(); i++) {
    while(idx <= transitions[i]) {
      vec[idvals[idx].id] = static_cast<T>(i);
      idx++;
    }
  }
  while (idx < idvals.size()) {
    vec[idvals[idx].id] = static_cast<T>(transitions.size());
    idx++;
  }
}

template<class T>
void check(const vector<T>& vec,
           const vector<double>& fvec,
           const vector<double>& transitions) {

  CHECK(vec.size() == fvec.size());
  for (int idx = 0; idx < vec.size(); idx++) {
    if (vec[idx] < transitions.size()) {
      CHECK(fvec[idx] <= transitions[vec[idx]])
        << "less or equal than transition! ";
    }

    if (vec[idx] > 0) {
      CHECK(fvec[idx] > transitions[vec[idx] - 1])
        << " larger than previous transition";
    }
  }
}

void check(const FeatureData& fd) {
  if (fd.encoding == BYTE) {
    check<uint8_t>(*(fd.bvec), *(fd.fvec), fd.transitions);
  } else if (fd.encoding == SHORT) {
    check<uint16_t>(*(fd.svec), *(fd.fvec), fd.transitions);
  }
}

void Bucketize(FeatureData& fd, bool useByteEncoding) {
  CHECK(fd.encoding == DOUBLE) << "invalid data to bucketing";

  const auto& fv = *(fd.fvec);
  const int num = fv.size();

  vector<IdVal> idvals;
  for (int i = 0; i < num; i++) {
    idvals.emplace_back(i, fv[i]);
  }

  sort(idvals.begin(), idvals.end(),
       [](const IdVal& x, const IdVal& y) {
         return x.val < y.val;
       });

  uint16_t maxValue
    = useByteEncoding ? numeric_limits<uint8_t>::max() : numeric_limits<uint16_t>::max();

  const int stepSize = ceil(fv.size()/(1.0 + maxValue));

  vector<int> transitions;
  int i = stepSize;
  while (i < num) {
    double t = idvals[i-1].val;
    while (i < num && idvals[i].val == t) {
      i++;
    }
    if (i < num) {
      transitions.push_back(i-1);
    }
    i += stepSize;
  }

  CHECK(transitions.size() < maxValue)
    << " invalid bucketing: too many buckets";

  for(int i = 0; i < transitions.size(); i++) {
    fd.transitions.push_back(idvals[transitions[i]].val);
  }

  bool byteEncoding = (transitions.size() < numeric_limits<uint8_t>::max());
  if (transitions.size() == 0) {
    fd.encoding = EMPTY;
  } else if (byteEncoding) {
    fd.encoding = BYTE;
    fd.bvec.reset(new vector<uint8_t>(num));
    fillValues<uint8_t>(idvals, transitions, *(fd.bvec));
  } else {
    fd.encoding = SHORT;
    fd.svec.reset(new vector<uint16_t>(num));
    fillValues<uint16_t>(idvals, transitions, *(fd.svec));
  }

  check(fd);

  // free up the original vector
  fd.fvec.reset();
}

void DataSet::bucketize() {
  if (!preBucketing_) {
    return;
  }

  LOG(INFO) << "start bucketization for data compression";
  int hist[4];
  memset(hist, 0, sizeof(hist));

  for (int i = 0; i < numFeatures_; i++) {
    Bucketize(features_[i], cfg_.isWeakFeature(i));
    hist[features_[i].encoding]++;

    LOG(INFO) << "feature: " << cfg_.getFeatureName(i)
              << " num transitions: " << features_[i].transitions.size()
              << ",encoding: " << features_[i].encoding;
  }
  preBucketing_ = false;
  CHECK(hist[3] == 0) << "no double features after bucketing";
  LOG(INFO) << "total memory saving over double: "
            << 1 - (hist[1] * 0.5 + hist[2])/(4.0*numFeatures_);
  LOG(INFO) << "additional memory saving over short: "
            << 1 - (hist[1] * 0.5 + hist[2])/numFeatures_;
}

}
