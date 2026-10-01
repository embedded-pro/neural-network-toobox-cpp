# Classification Metrics — Unit Test Plan (Pseudocode)

> GoogleTest · `TEST_F` (`float`) · `StrictMock` only · no heap.

## Fixture

```
class TestClassificationMetrics : public ::testing::Test:
    static constexpr std::size_t classes = 4
    static constexpr std::size_t samples = 10
    using ScoreVector = neural_network::ConfusionMatrix<float, classes>::ScoreVector
    const std::array<ScoreVector, samples> logits = {
        {  2.0,  0.1, -1.0,  0.0 },     # actual 0 → 0
        {  0.5,  1.5,  0.2, -0.3 },     # actual 1 → 1
        { -1.0,  0.3,  2.2,  0.1 },     # actual 2 → 2
        {  1.2,  1.1,  0.0,  0.0 },     # actual 1 → 0
        {  0.0,  0.0,  0.0,  0.9 },     # actual 2 → 3   (class 3 only ever predicted)
        {  3.0, -2.0,  0.5,  1.0 },     # actual 0 → 0
        { -0.5,  2.5,  2.4,  0.0 },     # actual 2 → 1
        {  0.1,  0.2,  0.3, -5.0 },     # actual 2 → 2
        {  0.7,  0.7,  0.1,  0.2 },     # actual 1 → 0   (exact tie ⇒ lowest index)
        { -2.0,  1.0,  0.0,  0.5 } }    # actual 1 → 1
    const std::array<std::size_t, samples> actual    = { 0, 1, 2, 1, 2, 0, 2, 2, 1, 1 }
    const std::array<std::size_t, samples> predicted = { 0, 1, 2, 0, 3, 0, 1, 2, 0, 1 }
    neural_network::ConfusionMatrix<float, classes> matrix
    neural_network::AccuracyCounter<float>          counter

    void AddAll():                                  # feeds both accumulators the same stream
        for s in 0..samples-1:
            matrix.AddScores(actual[s], logits[s])
            counter.Add(actual[s], predicted[s])
# each case below is a TEST_F(TestClassificationMetrics, <name>)
# ratios: EXPECT_NEAR(…, 1e-6f) (≈ 8 ulp at 1; distinct candidate formulas differ by ≥ 1e-2)
# counts and indices: EXPECT_EQ (exact integers)
# no collaborator has a virtual interface (free functions + value types) ⇒ no mock needed
# no Backward exists (argmax is piecewise constant) ⇒ no finite-difference gradient check
```

The fixture has every structural feature once: a perfectly recalled class (0), a perfectly precise class
(2), a class never predicted correctly and absent from `actual` (3, so `Recall(3)` is undefined while
`Precision(3)` and `F1(3)` are `0`), an exact score tie, and an unbalanced support `(2, 4, 4, 0)` that
makes `Accuracy`, `BalancedAccuracy` and `MacroF1` all differ.

## Test cases (Arrange / Act / Assert)

```
ArgMaxReturnsLowestIndexOfLargestScore:
    Act/Assert: EXPECT_EQ(ArgMax<float, 4>(v), index) for
        ( 2.0,  0.1, -1.0,  0.0) → 0      # first element is the max
        ( 0.0,  0.0,  0.0,  0.9) → 3      # last element (loop bound)
        (-3.0, -1.0, -2.0, -4.0) → 1      # all negative (catches a running max initialised to 0 instead of scores[0])
        (-0.5,  2.5,  2.4,  0.0) → 1      # close runner-up
        ( 0.7,  0.7,  0.1,  0.2) → 0      # tie at the max ⇒ lowest index
        ( 0.2,  0.9,  0.9,  0.9) → 1      # three-way tie not at index 0
        (-inf, -5.0, -inf, -7.0) → 1      # infinities are ordinary scores

BinaryLabelIsStrictlyGreaterThanThreshold:
    Assert: EXPECT_EQ(BinaryLabel(0.5f, 0.5f), 0)
            EXPECT_EQ(BinaryLabel(std::nextafter(0.5f, 1.0f), 0.5f), 1)    # 0.50000006
            EXPECT_EQ(BinaryLabel(0.49f, 0.5f), 0)
            EXPECT_EQ(BinaryLabel(0.0f, 0.0f), 0)                          # logit threshold
            EXPECT_EQ(BinaryLabel(1e-30f, 0.0f), 1)
            EXPECT_EQ(BinaryLabel(-1e-30f, 0.0f), 0)

AddScoresReturnsArgMaxAndCountsActualRowsPredictedColumns:
    Act:    for s in 0..9: returned[s] = matrix.AddScores(actual[s], logits[s])
    Assert: EXPECT_EQ(returned[s], predicted[s]) for every s
            EXPECT_EQ(matrix.Count(a, p), C[a][p]) for every (a, p), with
                C = [[2, 0, 0, 0],
                     [2, 2, 0, 0],
                     [0, 1, 2, 1],
                     [0, 0, 0, 0]]          # asymmetric: a row/column swap is caught (C[1][0] = 2, C[0][1] = 0)
            EXPECT_EQ(matrix.Total(), 10)

AccuracyIsCorrectOverTotalInBothAccumulators:
    Arrange: AddAll()
    Assert:  EXPECT_NEAR(*matrix.Accuracy(), 0.6f, 1e-6f)          # trace 6 / 10
             EXPECT_EQ(counter.Correct(), 6);  EXPECT_EQ(counter.Total(), 10)
             EXPECT_NEAR(*counter.Accuracy(), 0.6f, 1e-6f)

PerClassPrecisionRecallAndF1:
    Arrange: AddAll()
    Assert:  class k        : Precision        Recall          F1
             0              : 0.5              1.0             0.6666667   (2/3)
             1              : 0.6666667 (2/3)  0.5             0.5714286   (4/7)
             2              : 1.0              0.5             0.6666667   (2/3)
             3              : 0.0              nullopt         0.0
             EXPECT_NEAR(*matrix.Precision(k), P_k, 1e-6f) etc.;  EXPECT_FALSE(matrix.Recall(3).has_value())

BalancedAccuracyAndMacroF1SkipUndefinedClasses:
    Arrange: AddAll()
    Assert:  EXPECT_NEAR(*matrix.BalancedAccuracy(), 0.6666667f, 1e-6f)   # (1 + 1/2 + 1/2)/3, class 3 dropped
             EXPECT_NEAR(*matrix.MacroF1(), 0.4761905f, 1e-6f)            # (2/3 + 4/7 + 2/3 + 0)/4 = 10/21, class 3 kept

ResetClearsCountsAndMakesRatiosUndefined:
    Arrange: AddAll();  matrix.Reset();  counter.Reset()
    Assert:  EXPECT_EQ(matrix.Total(), 0);  EXPECT_EQ(matrix.Count(a, p), 0) for every (a, p)
             EXPECT_FALSE(matrix.Accuracy()); EXPECT_FALSE(matrix.BalancedAccuracy()); EXPECT_FALSE(matrix.MacroF1())
             EXPECT_FALSE(matrix.Precision(k)); EXPECT_FALSE(matrix.Recall(k)); EXPECT_FALSE(matrix.F1(k)) for every k
             EXPECT_EQ(counter.Total(), 0);  EXPECT_FALSE(counter.Accuracy())
    Act:     matrix.Add(1, 1);  counter.Add(1, 1)                            # reusable after Reset
    Assert:  EXPECT_NEAR(*matrix.Accuracy(), 1.0f, 1e-6f);  EXPECT_NEAR(*counter.Accuracy(), 1.0f, 1e-6f)
```

## Reference vectors

Computed by [`reference.py`](reference.py) (numpy `float32` scores, exact
`Fraction` arithmetic for the ratios, cross-checked against scikit-learn 1.9.1 `confusion_matrix`,
`accuracy_score`, `precision_score`/`recall_score`/`f1_score(average=None)`, `balanced_accuracy_score`
and `f1_score(average="macro")`):

| Quantity                                   | Value                                                             |
|--------------------------------------------|-------------------------------------------------------------------|
| predictions (lowest-index argmax)          | `(0, 1, 2, 0, 3, 0, 1, 2, 0, 1)` = `numpy.argmax`                 |
| confusion `C` (rows actual, cols predicted)| `[[2,0,0,0],[2,2,0,0],[0,1,2,1],[0,0,0,0]]`                       |
| support / predicted per class              | `(2, 4, 4, 0)` / `(4, 3, 2, 1)`                                   |
| accuracy                                   | `6/10 = 0.6` (`float32`: `0.6000000238`)                          |
| precision                                  | `(1/2, 2/3, 1, 0)`                                                |
| recall                                     | `(1, 1/2, 1/2, undefined)`                                        |
| F1 = `2TP/(support + predicted)`           | `(2/3, 4/7, 2/3, 0)`; equals `2PR/(P+R)` for classes 0–2          |
| balanced accuracy                          | `2/3 = 0.6666667` (sklearn drops class 3, warns)                  |
| macro F1                                   | `10/21 = 0.4761905` (sklearn default labels = union ⇒ 4 classes)  |
| argmax(softmax₃₂(z)) vs argmax(z), fixture | identical for all 10 rows                                         |
| `ArgMax` table                             | `0, 3, 1, 1, 0, 1, 1` (in the order listed above)                 |
| `BinaryLabel` table                        | `0, 1, 0, 0, 1, 0` (in the order listed above)                    |

## Edge cases

- **Exact ties** — lowest index, both directly (`ArgMax` table) and through `AddScores` (sample 9).
- **Undefined ratios** — `Recall(3)` in the fixture; every ratio after `Reset()`. `nullopt`, never NaN.
- **Class only predicted, never actual** — `Precision(3) = 0`, `F1(3) = 0`, counted in `MacroF1`,
  excluded from `BalancedAccuracy`.
- **Float softmax near-tie** (not unit-tested, it exercises `Softmax`, not this component):
  `z = (0, 1e-8)` ⇒ `float32` softmax `(0.5, 0.5)`, `ArgMax(softmax) = 0`, `ArgMax(z) = 1`; and
  `σ₃₂(1e-8) = 0.5` exactly. Documents why `ArgMax`/`BinaryLabel` should see logits.
- **Counter width** — `float32(2²⁴) + 1 == 2²⁴`, the reason counts are `uint32`. Not unit-tested:
  reaching it needs 16.8 M `Add` calls, too slow under QEMU.
- **Precondition asserts** (`actual/predicted/k ≥ K`, `total == UINT32_MAX`) — `really_assert`, not
  unit-tested, consistent with the other size/precondition asserts in the library.
- No separate balanced-data or two-class case: the fixture's unbalanced four-class stream already
  distinguishes every formula, and the `K = 2` path is the same code.
