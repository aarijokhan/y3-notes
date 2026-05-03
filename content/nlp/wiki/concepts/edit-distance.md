---
type: concept
sources:
  - raw/week-01/w01-slides.pdf
  - raw/week-01/w01-l03-transcript.txt
  - raw/week-01/w01-lab-02-solutions.pdf
  - raw/week-01/w01-lab-02-solutions-pt2.pdf
status: draft
updated: 2026-04-20
---

*Edit distance is the minimum number of single-character operations (insertion, deletion, substitution) needed to transform one string into another; it is computed exactly by dynamic programming in $O(nm)$ time.*

## Definition

The **edit distance** (or **Levenshtein distance**) between strings $X$ of length $n$ and $Y$ of length $m$ is the minimum-cost sequence of **edit operations** that transforms $X$ into $Y$:

- **Insertion**: insert a character into $X$
- **Deletion**: delete a character from $X$
- **Substitution**: replace a character in $X$ with a different character

In the standard Levenshtein metric, **insertion and deletion cost 1; substitution costs 2** (a substitution is modelled as a delete + insert). Some sources (including many lab/exam questions) use **uniform costs where substitution = 1**. Always check which convention a question uses — the same pair of strings will produce a different distance under the two conventions.

| Convention | ins | del | sub | Used by |
|---|---|---|---|---|
| Levenshtein (J&M) | 1 | 1 | **2** | Jurafsky & Martin textbook |
| Uniform | 1 | 1 | **1** | Most lab/exam problems, other textbooks |

> [!warning] Always check the convention before filling the table
> The diagonal cost is the only thing that differs between the two. A mismatch on the diagonal costs **+2** under Levenshtein and **+1** under uniform — the same pair of strings will give a different final distance depending on which you use.
> - The **`cat → cut`** worked example below uses **Levenshtein** (sub = 2)
> - The **`leda → deal`** worked example below uses **uniform** (sub = 1)

### Example: INTENTION → EXECUTION

```
I N T E N T I O N
| | | | | | | | |
* E X E C U T I O N
```

Edit sequence (cost 8): delete I (1), substitute N→E (2), substitute T→X (2), delete E (1), insert C (1), substitute N→U (2)... working through the alignment gives total cost 8 under Levenshtein.

## Dynamic Programming Algorithm

Define $D(i, j)$ = edit distance between $X[1..i]$ and $Y[1..j]$.

**Base cases**:
$$D(i, 0) = i \qquad D(0, j) = j$$

(Transforming any string to the empty string costs $i$ deletions; inserting $j$ characters into the empty string costs $j$ insertions.)

**Recurrence**:
$$D(i, j) = \min \begin{cases}
D(i-1, j) + 1 & \text{(deletion)} \\
D(i, j-1) + 1 & \text{(insertion)} \\
D(i-1, j-1) + \begin{cases} 0 & \text{if } X[i] = Y[j] \\ 2 & \text{if } X[i] \neq Y[j] \end{cases} & \text{(match / substitution)}
\end{cases}$$

The answer is $D(n, m)$.

**Complexity**: Time $O(nm)$, Space $O(nm)$ (can be reduced to $O(\min(n,m))$ if only the distance is needed, not the alignment).

### Worked example: `cat` → `cut`

$X$ = `cat` (rows), $Y$ = `cut` (columns). Levenshtein costs: sub = 2, ins/del = 1.

**Key reminder**: $D(i,j)$ = edit distance between the first $i$ chars of $X$ and the first $j$ chars of $Y$. You are building up the answer for the full strings from answers for all their prefixes.

**Step 1 — borders (base cases)**

Row 0: $D(0,j)=j$ — insert $j$ characters to build $Y[1..j]$ from nothing.
Column 0: $D(i,0)=i$ — delete $i$ characters to reduce $X[1..i]$ to nothing.

|      | `""`  | `c` | `u` | `t` |
| ---- | ----- | --- | --- | --- |
| `""` | **0** | 1   | 2   | 3   |
| `c`  | 1     | ?   | ?   | ?   |
| `a`  | 2     | ?   | ?   | ?   |
| `t`  | 3     | ?   | ?   | ?   |

**Step 2 — fill each interior cell**

For each cell, take the min of three options: **above+1** (delete), **left+1** (insert), **diagonal+0or2** (match/sub).

$D(1,1)$: `c` vs `c` — **match** → diagonal + 0 = $D(0,0)+0 = \mathbf{0}$

$D(1,2)$: `c` vs `u` — mismatch → min(above: $1+1=2$, left: $\mathbf{0+1=1}$, diag: $1+2=3$) = **1**

$D(1,3)$: `c` vs `t` — mismatch → min(above: $2+1=3$, left: $\mathbf{1+1=2}$, diag: $1+2=3$) = **2**

$D(2,1)$: `a` vs `c` — mismatch → min(above: $\mathbf{0+1=1}$, left: $2+1=3$, diag: $1+2=3$) = **1**

$D(2,2)$: `a` vs `u` — mismatch → min(above: $1+1=2$, left: $1+1=2$, diag: $0+2=\mathbf{2}$) = **2**

$D(2,3)$: `a` vs `t` — mismatch → min(above: $2+1=3$, left: $2+1=3$, diag: $1+2=3$) = **3**

$D(3,1)$: `t` vs `c` — mismatch → min(above: $\mathbf{1+1=2}$, left: $3+1=4$, diag: $2+2=4$) = **2**

$D(3,2)$: `t` vs `u` — mismatch → min(above: $2+1=3$, left: $2+1=3$, diag: $1+2=3$) = **3**

$D(3,3)$: `t` vs `t` — **match** → diagonal + 0 = $D(2,2)+0 = \mathbf{2}$

**Completed table:**

|  | `""` | `c` | `u` | `t` |
|---|---|---|---|---|
| `""` | 0 | 1 | 2 | 3 |
| `c` | 1 | **0** | 1 | 2 |
| `a` | 2 | 1 | 2 | 3 |
| `t` | 3 | 2 | 3 | **2** ← answer |

$D(3,3) = 2$: one substitution `a → u`, cost 2. ✓

### Visual summary: `SIT` → `SAT` (backtrace path in bold)

|       | **""** | **S** | **A** | **T** |
|-------|--------|-------|-------|-------|
| **""**| **0**  | 1     | 2     | 3     |
| **S** | 1      | **0** | 1     | 2     |
| **I** | 2      | 1     | **2** | 2     |
| **T** | 3      | 2     | 2     | **2** |

Reading the bold diagonal path top-left → bottom-right:
- (0,0)→(1,1): S = S → match, +0
- (1,1)→(2,2): I ≠ A → substitution, +2
- (2,2)→(3,3): T = T → match, +0

**Edit distance = 2.** Alignment: `S I T` → `S A T` (one substitution).

### Worked example: `leda` → `deal` (uniform costs, sub = 1)

This example uses the uniform cost variant (ins = del = sub = 1). $X$ = `leda` (rows), $Y$ = `deal` (columns).

**Base cases**: row 0 = 0,1,2,3,4; column 0 = 0,1,2,3,4.

**Interior cells** (each = min of above+1, left+1, diagonal+0or1):

$D(1,1)$: `l` vs `d` — mismatch → min(above: $1+1=2$, left: $1+1=2$, diag: $0+1=\mathbf{1}$) = **1**

$D(1,2)$: `l` vs `e` — mismatch → min(above: $2+1=3$, left: $\mathbf{1+1=2}$, diag: $1+1=2$) = **2**

$D(1,3)$: `l` vs `a` — mismatch → min(above: $3+1=4$, left: $\mathbf{2+1=3}$, diag: $2+1=3$) = **3**

$D(1,4)$: `l` vs `l` — **match** → min(above: $4+1=5$, left: $3+1=4$, diag: $3+0=\mathbf{3}$) = **3**

$D(2,1)$: `e` vs `d` — mismatch → min(above: $\mathbf{1+1=2}$, left: $2+1=3$, diag: $1+1=2$) = **2**

$D(2,2)$: `e` vs `e` — **match** → min(above: $2+1=3$, left: $2+1=3$, diag: $1+0=\mathbf{1}$) = **1**

$D(2,3)$: `e` vs `a` — mismatch → min(above: $3+1=4$, left: $\mathbf{1+1=2}$, diag: $2+1=3$) = **2**

$D(2,4)$: `e` vs `l` — mismatch → min(above: $3+1=4$, left: $\mathbf{2+1=3}$, diag: $3+1=4$) = **3**

$D(3,1)$: `d` vs `d` — **match** → min(above: $2+1=3$, left: $3+1=4$, diag: $2+0=\mathbf{2}$) = **2**

$D(3,2)$: `d` vs `e` — mismatch → min(above: $\mathbf{1+1=2}$, left: $2+1=3$, diag: $2+1=3$) = **2**

$D(3,3)$: `d` vs `a` — mismatch → min(above: $2+1=3$, left: $2+1=3$, diag: $\mathbf{1+1=2}$) = **2**

$D(3,4)$: `d` vs `l` — mismatch → min(above: $3+1=4$, left: $\mathbf{2+1=3}$, diag: $2+1=3$) = **3**

$D(4,1)$: `a` vs `d` — mismatch → min(above: $\mathbf{2+1=3}$, left: $4+1=5$, diag: $3+1=4$) = **3**

$D(4,2)$: `a` vs `e` — mismatch → min(above: $\mathbf{2+1=3}$, left: $3+1=4$, diag: $2+1=3$) = **3**

$D(4,3)$: `a` vs `a` — **match** → min(above: $2+1=3$, left: $3+1=4$, diag: $2+0=\mathbf{2}$) = **2**

$D(4,4)$: `a` vs `l` — mismatch → min(above: $3+1=4$, left: $\mathbf{2+1=3}$, diag: $2+1=3$) = **3**

**Completed table (sub = 1):**

|  | `""` | `d` | `e` | `a` | `l` |
|---|---|---|---|---|---|
| `""` | 0 | 1 | 2 | 3 | 4 |
| `l` | 1 | 1 | 2 | 3 | 3 |
| `e` | 2 | 2 | 1 | 2 | 3 |
| `d` | 3 | 2 | 2 | 2 | 3 |
| `a` | 4 | 3 | 3 | 2 | **3** ← answer |

**One optimal edit sequence** (total cost 3):
1. Substitute `l → d` (cost 1): `leda → deda`
2. Delete `d` (cost 1): `deda → dea`
3. Insert `l` at end (cost 1): `dea → deal`

> [!tip] Notice that `leda` and `deal` contain exactly the same four characters — yet the edit distance is 3, not 0. Edit distance measures the cost of transforming one string into another in order, not just character-set overlap.

## Backtrace

To recover the **alignment** (the actual edit sequence), store a pointer $\text{ptr}(i, j)$ alongside each $D(i, j)$:

- $\leftarrow$ (LEFT): came from $D(i, j-1)$ — insertion
- $\downarrow$ (DOWN): came from $D(i-1, j)$ — deletion
- $\searrow$ (DIAGONAL): came from $D(i-1, j-1)$ — match or substitution

Trace back from $D(n, m)$ to $D(0, 0)$. The path reads out the edit sequence in reverse; reversing gives the alignment.

Backtrace complexity: $O(n + m)$.

### Backtrace on `cat` → `cut`

At each cell, record which neighbour gave the minimum. Arrows below show the winning direction:

|  | `""` | `c` | `u` | `t` |
|---|---|---|---|---|
| `""` | 0 | ←1 | ←2 | ←3 |
| `c` | ↑1 | ↖**0** | ←1 | ←2 |
| `a` | ↑2 | ↑1 | ↖2 | ↖3 |
| `t` | ↑3 | ↑2 | ↑3 | ↖**2** |

Trace from $D(3,3)$: diagonal → $D(2,2)$ → diagonal → $D(1,1)$ → diagonal → $D(0,0)$.

Reading the diagonals forward: `c`=`c` (match), `a`≠`u` (substitute), `t`=`t` (match). Edit sequence: **substitute `a` → `u`**.

> [!tip] TIP — Three-step checklist for exam problems
> 1. **Write the strings as headers** — $X$ on rows, $Y$ on columns. Fill the `""` row and column with 0, 1, 2, … borders first.
> 2. **Fill left-to-right, top-to-bottom.** Each cell = min(above+1, left+1, diagonal+0or2). Check whether the characters match before computing the diagonal cost.
> 3. **Answer = bottom-right cell.** Backtrace = follow arrows back to (0,0); the diagonal steps are matches or substitutions, up = deletion, left = insertion.

## Weighted Edit Distance

The standard costs (ins=1, del=1, sub=2) treat all errors as equally likely. Real applications benefit from **weighted** costs:

$$D(i, j) = \min \begin{cases}
D(i-1, j) + \text{del}[X[i]] \\
D(i, j-1) + \text{ins}[Y[j]] \\
D(i-1, j-1) + \text{sub}[X[i], Y[j]]
\end{cases}$$

Motivation for different weights:
- **Spelling correction**: confused character pairs (e.g., `m/n`, `b/d`) should have lower substitution cost than unconfused pairs. Learned from spelling confusion matrices.
- **Keyboard proximity**: adjacent keys on a QWERTY keyboard (e.g., `s` and `d`) are more likely to be confused by a typist than distant keys.
- **OCR correction**: characters that look alike visually (`0/O`, `l/1`) should have lower substitution cost.

## Applications

- **Spell checking**: find the dictionary word with minimum edit distance to the misspelled word.
- **Machine translation evaluation**: BLEU and related metrics use edit distance ideas.
- **Computational biology**: sequence alignment (DNA, protein) uses a variant where costs reflect mutation probabilities.
- **Coreference and deduplication**: determining whether two name strings refer to the same entity.

## Related

- [[text-normalization]] — edit distance powers spell-checking, which is part of text normalization
- [[tokenization]] — edit distance can compare token sequences, not just character sequences

## Active Recall

> [!question]- State the recurrence relation for edit distance and explain what each of the three cases represents in terms of edit operations.
> $D(i,j) = \min\{D(i-1,j)+1,\ D(i,j-1)+1,\ D(i-1,j-1)+c\}$ where $c=0$ if $X[i]=Y[j]$ else $2$. The first case extends an alignment that aligned $X[1..i{-}1]$ with $Y[1..j]$ by deleting $X[i]$ from $X$ (cost 1). The second extends an alignment of $X[1..i]$ with $Y[1..j{-}1]$ by inserting $Y[j]$ into $X$ (cost 1). The third aligns $X[1..i{-}1]$ with $Y[1..j{-}1]$ and then either matches $X[i]$ to $Y[j]$ for free, or substitutes (cost 2).

> [!question]- How does the backtrace recover the actual alignment from the edit distance table, and what is its time complexity?
> Each cell stores a pointer indicating which predecessor cell it came from: LEFT (insertion), DOWN (deletion), or DIAGONAL (match/sub). Starting from $D(n,m)$, follow pointers back to $D(0,0)$. Each step reads off one edit operation. The path length is at most $n+m$ steps (the Manhattan distance in the grid), so backtrace is $O(n+m)$. Reversing the collected operations gives the alignment in forward order.

> [!question]- Why might you want weighted edit distance instead of uniform costs, and what data would you need to set the weights?
> Uniform costs assume all errors are equally likely, but in real applications some errors are far more common than others. For spell correction, character pairs that look or sound similar should have lower substitution cost (so the algorithm favours them as corrections). You would need a **confusion matrix** — empirical frequencies of which characters get confused for which, collected from a corpus of misspellings paired with their corrections, or from keyboard/OCR error logs.

> [!question]- Edit distance is $O(nm)$ in time and space. In what situation would this be prohibitively slow, and what simplification is available when you only need the distance (not the alignment)?
> For very long strings — e.g., comparing genomic sequences of millions of base pairs — $O(nm)$ space is prohibitive (the full table cannot fit in memory). When you only need the final distance $D(n,m)$ and not the alignment, you can use the two-row (or two-column) DP variant: at any point you only need the current and previous row, reducing space to $O(\min(n,m))$. Recovering the alignment still requires $O(nm)$ space (you must store the pointer table) or a divide-and-conquer approach.

> [!question]- Which of the following best describes the actions counted by minimum edit distance? A) Counting the number of words to add, delete, or replace in a text B) Calculating the number of characters that differ, without considering order C) Determining the minimum number of character insertions, deletions, and substitutions to change one string into another D) Measuring the total number of characters in two strings and finding the difference
> **C** is correct. Minimum edit distance operates at the character level and finds the minimum-cost sequence of insertions, deletions, and substitutions to transform one string into the other. **A** confuses character-level with word-level operations. **B** ignores order and position, which edit distance explicitly considers via the DP alignment. **D** just computes a length difference, which ignores the actual content of the strings entirely.

> [!question]- Using uniform costs (ins = del = sub = 1), is `drive` closer to `brief` or to `divers`? What is the distance to each?
> `drive → brief` = **3**, and `drive → divers` = **3**. They are equally close. This illustrates that edit distance comparisons can tie — you cannot always determine the closer word just by looking at length difference or shared characters. The strings share different subsets of characters with `drive`: `brief` shares r,i,e but differs in length; `divers` is one character longer but shares d,i,v,e,r in a compatible order. Working through both DP tables is the only reliable way to settle such comparisons.
