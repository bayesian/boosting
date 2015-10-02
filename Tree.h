#pragma once

#include <boost/scoped_array.hpp>

#include "folly/json.h"
#include "folly/Conv.h"

namespace boosting {

template <class T>
class TreeNode {
 public:
  virtual double eval(const boost::scoped_array<T>& fvec) const = 0;
  virtual void scale(double w) = 0;
  virtual folly::dynamic toJson() const = 0;
  virtual ~TreeNode() {}
};

template <class T>
class PartitionNode : public TreeNode<T> {
 public:
  PartitionNode(int fid, T fv)
    : fid_(fid), fv_(fv),
    left_(NULL), right_(NULL) {
  }

  TreeNode<T>* getLeft() const {
    return left_;
  }

  TreeNode<T>* getRight() const {
    return right_;
  }

  int getFid() const {
    return fid_;
  }

  T getFv() const {
    return fv_;
  }

  void setLeft(TreeNode<T>* left) {
    left_ = left;
  }

  void setRight(TreeNode<T>* right) {
    right_ = right;
  }

  double eval(const boost::scoped_array<T>& fvec) const {
    if (fvec[fid_] <= fv_) {
      return left_->eval(fvec);
    } else {
      return right_->eval(fvec);
    }
  }

  void scale(double w) {
    left_->scale(w);
    right_->scale(w);
  }

  folly::dynamic toJson() const {
    folly::dynamic m = folly::dynamic::object;

    m.insert("index", fid_);
    m.insert("value", fv_);
    m.insert("left", left_->toJson());
    m.insert("right", right_->toJson());
    return m;
  }

  ~PartitionNode() {
    delete left_;
    delete right_;
  }

 private:
  int fid_;
  T fv_;
  TreeNode<T>* left_;
  TreeNode<T>* right_;
};

template <class T>
class LeafNode : public TreeNode<T> {
 public:
  explicit LeafNode(double fvote) : fvote_(fvote) {
  }

  double eval(const boost::scoped_array<T>& fvec) const {
    return fvote_;
  }

  double getVote() const {
    return fvote_;
  }

  void scale(double w) {
    fvote_ *= w;
  }

  folly::dynamic toJson() const {
    folly::dynamic m = folly::dynamic::object;

    m.insert("index", -1);
    m.insert("value",  fvote_);
    return m;
  }

  ~LeafNode() {
  }

 private:
  double fvote_;
};

// load a regression tree from Json
template <class T>
TreeNode<T>* fromJson(const folly::dynamic& obj) {
  int index = obj["index"].asInt();

  T v;
  if (obj["value"].isInt()) {
    v = static_cast<T>(obj["value"].asInt());
  } else {
    v = static_cast<T>(obj["value"].asDouble());
  }

  if (index == -1) {
    return new LeafNode<T>(v);
  } else {
    PartitionNode<T>* rt = new PartitionNode<T>(index, v);
    rt->setLeft(fromJson<T>(obj["left"]));
    rt->setRight(fromJson<T>(obj["right"]));
    return rt;
  }
}

template <class T>
  double predict(const std::vector<TreeNode<T>*>& models,
                 const boost::scoped_array<T>& fvec) {

  double f = 0.0;
  for (const auto& m : models) {
    f += m->eval(fvec);
  }
  return f;
}

template <class T>
  double predict_vec(const std::vector<TreeNode<T>*>& models,
                     const boost::scoped_array<T>& fvec,
                     std::vector<double>* score) {

  double f = 0.0;
  for (const auto& m : models) {
    f += m->eval(fvec);
    score->push_back(f);
  }
  return f;
}


}
