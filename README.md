# Transformer Puzzles

<a target="_blank" href="https://colab.research.google.com/github/srush/Transformer-Puzzles/blob/main/TransformerPuzzlers.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>



<!-- #region id="e9e822cb" -->
This notebook is a collection of short coding puzzles based on the internals of the Transformer. The puzzles are written in Python and can be done in this notebook. After completing these you will have a much better intutive sense of how a Transformer can compute certain logical operations. 

These puzzles are based on [Thinking Like Transformers](https://arxiv.org/pdf/2106.06981.pdf) by Gail Weiss, Yoav Goldberg, Eran Yahav and derived from this [blog post](https://srush.github.io/raspy/).
<!-- #endregion -->

![image](https://user-images.githubusercontent.com/35882/235678934-44c83052-9743-4de7-a46c-49a517923da1.png)


<!-- #region id="8e962052" -->
## Goal

**Can we produce a Transformer that does basic elementary school addition?**

i.e. given a string "19492+23919" can we produce the correct output? 
<!-- #endregion -->

<!-- #region id="d332140b" -->
## Rules

Each exercise consists of a function with a argument `seq` and output `seq`. Like a transformer we cannot change length. Operations need to act on the entire sequence in parallel. There is a global `indices` which tells use the position in the sequence. If we want to do something different on certain positions we can use `where` like in Numpy or PyTorch. To run the seq we need to give it an initial input. 
<!-- #endregion -->


```python colab={"base_uri": "https://localhost:8080/", "height": 96} id="1b28dc98" outputId="f1ac1157-3db8-40c0-dbb2-7d9bad8943a0"
def even_vals(seq=tokens):
    "Keep even positions, set odd positions to -1"
    x = indices % 2
    # Note that all operations broadcast so you can use scalars.
    return where(x == 0, seq, -1)
seq = even_vals()

# Give the initial input tokens
seq.input([0,1,2,3,4])
```

<!-- #region id="9dc23f88" -->
The main operation you can use is "attention". You do this by defining a selector which forms a matrix based on `key` and `query`.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 176} id="e2ee0ff8" outputId="a61ac19c-2550-4f3c-d653-50c323cdfd59"
before = key(indices) < query(indices)
before
```

<!-- #region id="a4de0a14" -->
We can combine selectors with logical operations.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 201} id="c315ba6d" outputId="270d50fa-649c-438b-8606-d3d078478162"
before_or_same = before | (key(indices) == query(indices))
before_or_same
```

<!-- #region id="00bc66a3" -->
Once you have a selector, you can apply "attention" to sum over the grey positions. For example to compute cumulative such we run the following function. 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 326} id="e79c8c8b" outputId="44db7f90-502d-497c-c5ba-4062c09f0a9a"
def cumsum(seq=tokens):
    return before_or_same.value(seq)
seq = cumsum()
seq.input([0, 1, 2, 3, 4])
```

Good luck!
