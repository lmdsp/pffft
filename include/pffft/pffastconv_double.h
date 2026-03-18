/* Copyright (c) 2019  Hayati Ayguen ( h_ayguen@web.de )

   Redistribution and use of the Software in source and binary forms,
   with or without modification, is permitted provided that the
   following conditions are met:

   - Neither the names of PFFFT, PFFASTCONV, nor the names of its
   sponsors or contributors may be used to endorse or promote products
   derived from this Software without specific prior written permission.  

   - Redistributions of source code must retain the above copyright
   notices, this list of conditions, and the disclaimer below.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions, and the disclaimer below in the
   documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
   HOLDERS BE LIABLE FOR ANY CLAIM, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
   SOFTWARE.
*/

/*
   PFFASTCONV : a Pretty Fast Fast Convolution

   This is basically the implementation of fast convolution,
   utilizing the FFT (pffft).

   Restrictions: 

   - 1D transforms only, with 64-bit double precision.

   - all (double*) pointers in the functions below are expected to
   have an "simd-compatible" alignment, that is 16 bytes on x86 and
   powerpc CPUs.
  
   You can allocate such buffers with the functions
   pffft_aligned_malloc / pffft_aligned_free (or with stuff like
   posix_memalign..)

*/

#ifndef PFFASTCONV_DOUBLE_H
#define PFFASTCONV_DOUBLE_H

#include <stddef.h> /* for size_t */
#include "pffft.h"


#ifdef __cplusplus
extern "C" {
#endif

  /* opaque struct holding internal stuff
     this struct can't be shared by many threads as it contains
     temporary data, computed within the convolution
  */
  typedef struct PFFASTCONVD_Setup PFFASTCONVD_Setup;

  /*
    prepare for performing fast convolution(s) of 'filterLen' with input 'blockLen'.
    The output 'blockLen' might be bigger to allow the fast convolution.
    
    'flags' are bitmask over the 'pffastconv_flags_t' enum.

    PFFASTCONV_Setup structure can't be shared accross multiple filters
    or concurrent threads.
  */
  PFFASTCONVD_Setup* pffastconvd_new_setup( const double * filterCoeffs, int filterLen, int * blockLen, int flags );

  void pffastconvd_destroy_setup(PFFASTCONVD_Setup*);

  /* 
     Perform the fast convolution.

     'input' and 'output' don't need to be aligned - unless any of
     PFFASTCONV_DIRECT_INP or PFFASTCONV_DIRECT_OUT is set in 'flags'.

     inputLen > output 'blockLen' (from pffastconv_new_setup()) is allowed.
     in this case, multiple FFTs are called internally, to process the
     input[].

     'output' vector must have size >= (inputLen - filterLen + 1)

     set bool option 'applyFlush' to process the full input[].
     with this option, 'tail samples' of input are also processed.
     This might be inefficient, because the FFT is called to produce
     few(er) output samples, than possible.
     This option is useful to process the last samples of an input (file)
     or to reduce latency.

     return value is the number of produced samples in output[].
     the same amount of samples is processed from input[]. to continue
     processing, the caller must save/move the remaining samples of
     input[].

  */
  int pffastconvd_apply(PFFASTCONVD_Setup* s, const double *input, int inputLen, double *output, int applyFlush);

  void *pffastconvd_malloc(size_t nb_bytes);
  void pffastconvd_free(void *);

  /* return 4 or 1 wether support SSE/Altivec instructions was enabled when building pffft.c */
  int pffastconvd_simd_size();


#ifdef __cplusplus
}
#endif

#endif /* PFFASTCONV_DOUBLE_H */
