Mihail Eric -- meric
Raymond Wu -- wur911

Our code uses OpenMP #pragma directives to parallelize a lot of the nested for-loop computations that occur for computing the fourier and inverse fourier transform. For example, in each cpu_(i)fftx/y method, we added a '#pragma omp parallel' and a subsequent '#pragma omp for' before the outermost for-loop of the computations. Since these computations have triple-nested for-loops we privatize the index variables of each for-loop, because each for-loop performs an independent computation that can be done in parallel, withouth relying on previous iterations of the loop. In order to allow for sharing of all the real and imaginary component buffers, we changed these buffers to be statically allocated inside of the '#pragma omp' loops instead of being dynamically allocated.

Given these OpenMP parallelizations, we found that we achieved a roughly 5.8x speedup on the large image. Testing for correctness, we found that the image quality of all the images degraded a bit, but all the images were still recognizable.
