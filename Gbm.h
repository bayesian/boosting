#pragma once

#include <cstdint>
#include <vector>

namespace boosting {

class Config;
class DataSet;
class GbmFun;

template<class T> class TreeNode;

class Gbm {
 public:
  Gbm(const GbmFun& fun,
      const DataSet& ds,
      const Config& cfg);

  void getModel(std::vector<TreeNode<double>*>* model,
                double fimps[]);

 private:

  TreeNode<double>* mapTree(const TreeNode<uint16_t>* rt);

  const GbmFun& fun_;
  const DataSet& ds_;
  const Config& cfg_;
};

}
