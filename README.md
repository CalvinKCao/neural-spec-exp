# Minimal NAP specifications with a cheaper verifier

*Disclaimer: I gave an LLM my bullet points and had it write the first draft. The majority of the below final text is my own.*

Geng et al, "Learning Minimal Neural Specifications," describe a way to
specify what a class looks like to a ReLU network. For a given input, each
hidden neuron is either on or off. If almost all training examples of a class
share the on/off state of a neuron, that state goes into a pattern for the
class. If a verifier proves every input matching the pattern gets
classified as that class, the pattern works as a specification of the class.

The paper then shrinks the pattern by releasing neurons one at a time and
asking the verifier whether the guarantee still holds. What is left is called
minimal, but only relative to the verifier doing the checking. The paper uses
an exact solver, which is slow and limits the method to small networks.

If a much cheaper verifier gives similar patterns, the method could scale further. 

## What I did

I replaced the exact solver with a linear programming relaxation. It cannot prove
everything the exact solver can. I ran the same shrinking procedure with both
on a small network trained on the breast cancer dataset (32 neurons) across five seeds and both classes. I also checked the exact solver against a second independent solver on one run, and they agreed on every query both finished.

## Results

- The LP is much cheaper. Roughly a hundred times faster per call, and the
  patterns it produces are a small fraction of the mined pattern. They cover
  more own-class test examples than the mined pattern and rarely cover
  other-class ones. In my runs the pinned neurons were all in the last hidden
  layer.
- It isn't as strong. In some cases the mined pattern was provable by the
  exact solver but not by the relaxation.
- Release order matters a lot. With the LP, the final pattern ranged from
  under ten neurons to most of the mined pattern depending on the order.
  Ordering neurons by LP dual values ended at the same result as simple
  forward order, so it added nothing.

## Limitations

The exact solver hit its 5 minute limit on every neuron it kept, so its
pattern sizes are upper bounds, and I can't say which neurons are truly
essential. I couldn't compare the two verifiers "minimal" patterns directly.
Test sets are small and there are only five seeds. As such, it's weak evidence. This is one small network and one dataset.

## Next steps

Try a bound-propagation verifier (e.g. alpha,beta-CROWN) and longer solver limits.
