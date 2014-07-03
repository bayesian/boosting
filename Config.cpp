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

    const dynamic& trainColumns = cfg["train_columns"];
    for (auto it = trainColumns.begin(); it != trainColumns.end(); ++it) {
      trainIdx_.push_back(columnIdx.at(it->asString()));
    }

    const dynamic& weakColumns = cfg["weak_columns"];
    for (auto it = weakColumns.begin(); it != weakColumns.end(); ++it) {
      weakIdx_.push_back(columnIdx.at(it->asString()));
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
