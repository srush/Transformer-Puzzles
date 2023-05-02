# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + tags=["hide_inp"]
# !pip install -qqq git+https://github.com/chalk-diagrams/chalk git+https://github.com/srush/RASPy 

# +
from raspy.rasp import key, query, tokens, indices, where


# -

# # Transformer Puzzles

# Based on [Thinking Like Transformers](https://arxiv.org/pdf/2106.06981.pdf) by Gail Weiss, Yoav Goldberg, Eran Yahav

# Transformer models are foundational to AI systems. There are now countless explanations of "how transformers work?" in the sense of the architecture diagram at the heart of transformers.

# However this diagram does not provide any intuition into the computational model of this framework. As researchers become interested in how Transformers work, gaining intuition into their mechanisms becomes increasingly useful.

# [Thinking like Transformers](https://arxiv.org/pdf/2106.06981.pdf) proposes a computional framework for  Transformer-like calculations. The framework uses discrete computation to simulate Transformer computations.  The resulting language [RASP](https://github.com/tech-srl/RASP) is a programming language where every program compiles down to a specific Transformer.

# In this blog post, I reimplemented a variant of RASP in Python ([RASPy](https://github.com/srush/raspy)). The language is roughly compatible with the original version, but with some syntactic changes that I thought were fun. With this language, the author of the work Gail Weiss, provided a challenging set of puzzles to walk through and understand how it works. 

# Before jumping into the language itself, let's look at an example of what coding with Transformers looks like. Here is some code that computes the `flip`, i.e. reversing an input sequence. The code itself uses two Transformer layers to apply attention and mathematical computations to achieve the result.

# Given this library of functions, we can write operations to accomplish surprisingly complex tasks. 
#
# Gail Weiss, the author of the paper, gave me a really challenging problem broken up into steps. 
#
# **Can we produce a Transformer that does basic addition?**
#
# i.e. given a string "19492+23919" can we produce the correct output? 

# If you would rather do these on your own, we provide [a version]() with this part of the notebook blank out.  

def cumsum(seq=tokens):
    before = key(indices) < query(indices)
    x = (before | (key(indices) == query(indices))).value(seq)
    return x.name("cumsum")

def atoi(seq=tokens):
    return seq.map(lambda x: ord(x) - ord('0'))


# We provide a few helper functions to make it easier to write transforms. For example, `where` provides an "if" statement like construct

where((tokens == "h") | (tokens == "l"), tokens, "q")


# ### Challenge 1: Select a given index
#
# Produce a sequence where all the elements have the value at index i.

def test_output(user, spec, token_sets):
    for token_set in token_sets:        
        out1 = user(*token_set[:-1])(token_set[-1]).toseq()
        out2 = spec(*token_set)
        for i, o in enumerate(out2):
            assert out1[i] == o, f"Output: {out1} Expected: {out2}"

SEQ = [2,1,3,2,4]
SEQ2 = [3, 4 ,3,-1,2]
            

def index_spec(i, seq):
    return [seq[i] for _ in seq]

def index(i, seq=tokens):
    x = (key(indices) == query(i)).value(seq)
    return x.name("index")

test_output(index, index_spec,
            [(2, SEQ),
             (3, SEQ2),
             (1, SEQ)])

# ### Challenge 2: Shift
#
# Shift all of the tokens in a sequence to the right by `i` positions.

def shift_spec(i, default="0", seq=None):
    return [default]*i + [s for j, s in enumerate(seq) if j < len(seq) - i]

def shift(i, default="0", seq=tokens):
    x = (key(indices) == query(indices-i)).value(seq, default)
    return x.name("shift")


test_output(shift, shift_spec,
            [(2, 0, SEQ),
             (3, 0, SEQ2),
             (1, 0, SEQ)])


# ### Challenge 3: Right Align
#
# Right align a padded sequence e.g. ralign().inputs('xyz___') = '000xyz'" (3 layers)


def ralign_spec(ldefault="0", seq=tokens):
    last = None
    for i in range(len(seq)-1, -1, -1):
        if seq[i] == "_":
            last = i
        else:
            break
    if last == None:
        return seq
    return [ldefault] * (len(seq) - last)  + seq[:last]

def ralign(ldefault="0", seq=tokens):
    c = (key(seq) == query("_")).value(1)
    x = (key(indices + c) == query(indices)).value(seq, ldefault)
    return x.name("ralign")

test_output(ralign, ralign_spec,
            [("-", list("xyzabc__"),), ("0", list("xyz___"),)])


# ### Challenge 4: Split
#
# Split a sequence on a value. Get the first or second part. Right align. (5 layers)

def split_spec(v, get_first_part, seq):
    out = []
    mid = False
    blank = "0" if not get_first_part else "_"
    for j, s in enumerate(seq):
        if s == v:
            out.append(blank)
            mid = True
        elif (get_first_part and not mid) or (not get_first_part and mid):
            out.append(s)
        else:
            out.append(blank)
    return ralign_spec("0", seq=out)

def split(v, get_first_part, seq=tokens):
    split_point = (key(seq) == query(v)).value(indices)
    if get_first_part:
        x = ralign("0", seq=where(indices < split_point, 
                             seq, "_"))
        return x
    else:
        x = where(indices > split_point, seq, "0")
        return x

test_output(split, split_spec,
            [("-", 1, list("xyz-ax"),),
             ("-", 0, list("xyz-ax"),),
             ("+", 0, list("xy+z-ax"),)]
             )



# ### Challenge 5: Minimum 
#
# Compute the minimum values of the sequence. (This one starts to get harder. Our version uses 2 layers of attention.)

def minimum_spec(seq):
    m = min(seq)
    return [m for _ in seq]

def minimum(seq=tokens):
    before = key(indices) < query(indices)
    sel1 = before & (key(seq) == query(seq))
    sel2 = key(seq) < query(seq)
    less = (sel1 | sel2).value(1)
    x = (key(less) == query(0)).value(seq)
    return x.name("min")

test_output(minimum, minimum_spec,
            [(SEQ,), (SEQ2,)])

# ### Challenge 6: First Index
#
# Compute the first index that has token `token`. (2 layers)

def first_spec(token, seq):
    first = None
    for i, s in enumerate(seq):
        if s == token:
            first = i
    return [first for _ in seq]

def first(token, seq=tokens):
    return minimum(where(seq == token, indices, 99))

test_output(first, first_spec,
            [(3, SEQ), (-1, SEQ2)])



# ### Challenge 7: Slide
#
# Replace special tokens "<" with the closest non "<" value to their right. (2 layers)

def slide_spec(match, seq):
    out = []
    for i, s in enumerate(seq):
        if s == "<":
            for v in seq[i+1:]:
                if v != "<":
                    out.append(v)
                    break
        else:
            out.append(s)
    return out


def slide(match="<", seq=tokens):
    match = seq != "<" 
    x = cumsum(match) 
    y = ((key(x) == query(x + 1)) & (key(match) == query(True))).value(seq)
    seq =  where(match, seq, y)
    return seq.name("slide")

test_output(slide, slide_spec,
            [("<",  list("1<<2"),),
             ("<",  list("2<<<3"),),
             ("<",  list("3<<<1<<3"),)]
             )

# ### Challenge 7: Add
#
# For this one you want to perform addition of two numbers. Here are the steps. 
#
# add().input("683+345")
#
# 0) Split into parts. Convert to ints. Add
#
# > "683+345" => [0, 0, 0, 9, 12, 8]
#
# 1) Compute the carry terms. Three possibilities: 1 has carry, 0 no carry, < maybe has carry. 
#
# > [0, 0, 0, 9, 12, 8] => "00<100"
#
# 2) Slide the carry coefficients
#
# > "00<100" => 001100"
#
# 3) Complete the addition.
#
# Each of these is 1 line of code. The full system is 8 attentions.
#
#

def add_spec(seq):
    a, b = "".join(seq).split("+")
    c = int(a) + int(b)
    out = f"{c}"
    return list(map(int, list(("0" * (len(seq) - len(out))) + out)))


def add(seq=tokens):
    x = atoi(split("+", True, seq)) \
        + atoi(split("+", False, seq))
    # 1) Check for carries 
    gets_carry = shift(-1, "0", where(x > 9, "1", where(x == 9, "<", "0")))
    # 2) Slide carries to their columns - all in one parallel go!                                         
    gets_carry = atoi(slide("<", gets_carry))
    # 3) Add in carries, and remove overflow from original addition.                                                                               
    return (x + gets_carry) % 10


test_output(add, add_spec,
            [(list("1+2"),),
             (list("22+38"),),
             (list("3+10"),)]
             )
