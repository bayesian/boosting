#include "Gbm.h"

#include <boost/scoped_array.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

#include "Concurrency.h"
#include "Config.h"
#include "DataSet.h"
#include "GbmFun.h"
#include "Tree.h"
#include "TreeRegressor.h"
#include <gflags/gflags.h>

namespace boosting {

using namespace std;

Gbm::Gbm(const GbmFun& fun, const DataSet& ds, const Config& cfg)
  : fun_(fun), ds_(ds), cfg_(cfg) {
}

class ParallelEval : public apache::thrift::concurrency::Runnable {
 public:
  ParallelEval(
    CounterMonitor& monitor,
    const int numExamples,
    const int numFeatures,
    const GbmFun& fun,
    const std::unique_ptr<TreeNode<uint16_t>>& weakModel,
    const DataSet& ds,
    const vector<double>& targets,
    boost::scoped_array<double>& F,
    boost::scoped_array<double>& subLoss,
    const int workIdx,
    const int totalWorkers)
    : monitor_(monitor), numExamples_(numExamples),
      numFeatures_(numFeatures), fun_(fun), weakModel_(weakModel),
      ds_(ds), targets_(targets), F_(F),
      subLoss_(subLoss), workIdx_(workIdx),
      totalWorkers_(totalWorkers) {
  }

  void run() {
    //boost::scoped_array<uint16_t> fvec(new uint16_t[numFeatures_]);
    for (int i = 0; i < numExamples_; i++) {
      if (i % totalWorkers_ == workIdx_) {
        //ds_.getFeatureVec(i, fvec);
        //double score = weakModel_->eval(fvec);
        double score = ds_.getPrediction(weakModel_.get(), i);
        F_[i] += score;
        const double wt = ds_.getWeights() ? (*(ds_.getWeights()))[i] : 1.0;
        subLoss_[workIdx_] += fun_.getExampleLoss(targets_[i], F_[i], wt);
      }
    }
    monitor_.decrement();
  }

 private:
  CounterMonitor& monitor_;
  const int numExamples_;
  const int numFeatures_;
  const GbmFun& fun_;
  const std::unique_ptr<TreeNode<uint16_t>>& weakModel_;
  const DataSet& ds_;
  const vector<double> targets_;
  boost::scoped_array<double>& F_;
  boost::scoped_array<double>& subLoss_;
  const int workIdx_;
  const int totalWorkers_;
};

void Gbm::getModel(
  vector<TreeNode<double>*>* model,
  double fimps[]) {

  const int numExamples = ds_.getNumExamples();

  boost::scoped_array<double> F(new double[numExamples]);
  boost::scoped_array<double> y(new double[numExamples]);

  double f0 = fun_.getF0(ds_.targets_, ds_.getWeights().get());
  for (int i = 0; i < numExamples; i++) {
    F[i] = f0;
  }

  model->push_back(new LeafNode<double>(f0));

  double initLoss = fun_.getInitLoss(ds_.targets_, ds_.getWeights().get());

  LOG(INFO) << "init avg loss " << initLoss / numExamples;

  for (int it = 0; it < cfg_.getNumTrees(); it++) {

    LOG(INFO) << "------- iteration " << it << " -------";

    fun_.getGradient(ds_.targets_, F, y, ds_.getWeights().get());
    TreeRegressor regressor(ds_, y, fun_);

    std::unique_ptr<TreeNode<uint16_t>> weakModel(
      regressor.getTree(cfg_.getNumLeaves(), cfg_.getExampleSamplingRate(),
                        cfg_.getFeatureSamplingRate(), fimps));

    weakModel->scale(cfg_.getLearningRate());

    model->push_back(mapTree(weakModel.get()));

    VLOG(1) << toPrettyJson(weakModel->toJson(cfg_));
    double newLoss = 0.0;

    if (FLAGS_num_threads > 1) {
      CounterMonitor monitor(FLAGS_num_threads);
      boost::scoped_array<double> subLoss(new double[FLAGS_num_threads]);
      for (int wid = 0; wid < FLAGS_num_threads; wid++) {
        subLoss[wid] = 0.0;
        Concurrency::threadManager->add(
          boost::shared_ptr<apache::thrift::concurrency::Runnable>(
            new ParallelEval(monitor, numExamples, ds_.numFeatures_,
                             fun_, weakModel,
                             ds_, ds_.targets_, F, subLoss,
                             wid, FLAGS_num_threads)));
      }
      monitor.wait();

      for (int wid = 0; wid < FLAGS_num_threads; wid++) {
        newLoss += subLoss[wid];
      }
    } else {
      //boost::scoped_array<uint16_t> fvec(new uint16_t[ds_.numFeatures_]);
      for (int i = 0; i < numExamples; i++) {
        // ds_.getFeatureVec(i, fvec);
        // double score = weakModel->eval(fvec);
        double score = ds_.getPrediction(weakModel.get(), i);
        F[i] += score;
        const double wt = ds_.getWeights() ? (*(ds_.getWeights()))[i] : 1.0;
        newLoss += fun_.getExampleLoss(ds_.targets_[i], F[i], wt);
      }
    }

    LOG(INFO) << "total avg loss " << newLoss/numExamples
              << " reduction: " << 1.0 - newLoss/initLoss;
  }
}

TreeNode<double>* Gbm::mapTree(const TreeNode<uint16_t>* rt) {
  const PartitionNode<uint16_t>* pnode =
    dynamic_cast<const PartitionNode<uint16_t>*>(rt);
  if (pnode != NULL) {
    int fid = pnode->getFid();
    PartitionNode<double>* newNode = new PartitionNode<double>(
      fid, ds_.features_[fid].transitions[pnode->getFv()]);
    newNode->setVote(pnode->getVote());
    newNode->setLeft(mapTree(pnode->getLeft()));
    newNode->setRight(mapTree(pnode->getRight()));
    return newNode;
  } else {
    const LeafNode<uint16_t>* lfnode =
      dynamic_cast<const LeafNode<uint16_t>*>(rt);
    return new LeafNode<double>(lfnode->getVote());
  }
}

}
