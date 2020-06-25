

In the paper of Xnor-net and paper of BNN, a extremely quantization method which both activation and weight are binarized. 

They enable the possibility of substitute the muliplcation and acculumation opertions with bit-wise operations. Profiling on real plaforms shows a one magnitude order speedup is possible by `xnor` and `popcount` operations.

Though the initial performance in the paper seems a sizable gap from the full precision counterpart, it inspires a lot of effort in performance improving.
