Fast & Simple implementation of GBM

GBM is the generally regarded as best perform supervised learning algorithms before recent DL revolution. It is robust but not scalable.
 
Goal:
1) Fast (Handle 40M rows * 500 features within 10 hours)
2) Simple (The less lines of code, the better) <= 3000
3) Mudular/Extensible for further improvements

Algorithms:
1) pre-bucketing (data compression)
2) bucket sort to build histogram, then linear scan to find best split
3) hints and intelligent of using #buckets
4) stochastic gradient boosting machine

features:
1) correctness (model + fimps)
2) deterministic randomness
3) easily extensible for wide varieties of similar algorithms: random forest, bagging, gbm, for both classification and regression methods, regression takes priority

new features:
1) byte/short: two layer of storage. (save both memory and cpu)
2) taking hints based on previous fimps (top 1/3 using short, rest using byte)

Prameters:

m: number trees
n: number of leaves per tree
r: example sampling rate
s: feature sampling rate

d: number of data points
f: number of features

k: number of buckets
ml: minimum number of datapoints per leave

Complexity:
Memory: max(f * d1 * 8, [f * d, f * d * 2))

Algorithmic:
1. Bucketization: O(f * d1 * log(d1))
2. Continue reading: O(f * d2 * log(k))

3: Single Best Split: O(f' * d' + f' * k)
4a: depth-k balanced tree: k * S
4b: single n-leaves tree: #splits: (2n - 3), O(S * n * log(n)) (roughly)

D: 20M, exampling sampling: 4M
feature sampling rate:

Components:

Config:        (specify data format and training parameters)
DataSet:       (column-wise storage, with Self Compression)
Tree:          (works both in compressed/raw)
TreeRegressor: (k-leaf regression tree)
GbmFun:        (function to extend to different types of loss)
Gbm:           (gradient boosting machine)

