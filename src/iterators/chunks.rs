use crate::imp_prelude::*;
use crate::iter::ExactChunksIter;
use crate::IntoDimension;
use crate::iter::ExactChunksIterMut;
use crate::{Layout, NdProducer};

use super::BIItemArrayViewInner;

impl_ndproducer! {
    ['a, A, D: Dimension]
    [Clone => 'a, A, D: Clone ]
    ExactChunks {
        base,
        chunk,
        inner_strides,
    }
    ExactChunks<'a, A, D> {
        type Item = ArrayView<'a, A, D>;
        type Dim = D;

        unsafe fn item(&self, ptr) {
            ArrayView::new_(ptr, self.chunk.clone(),
                            self.inner_strides.clone())
        }
    }
}

type BaseProducerRef<'a, A, D> = ArrayView<'a, A, D>;
type BaseProducerMut<'a, A, D> = ArrayViewMut<'a, A, D>;

/// Exact chunks producer and iterable.
///
/// See [`.exact_chunks()`](ArrayBase::exact_chunks) for more
/// information.
//#[derive(Debug)]
pub struct ExactChunks<'a, A, D> {
    base: BaseProducerRef<'a, A, D>,
    chunk: D,
    inner_strides: D,
}

impl<'a, A, D: Dimension> ExactChunks<'a, A, D> {
    /// Creates a new exact chunks producer.
    ///
    /// **Panics** if any chunk dimension is zero
    pub(crate) fn new<E>(mut a: ArrayView<'a, A, D>, chunk: E) -> Self
    where
        E: IntoDimension<Dim = D>,
    {
        let chunk = chunk.into_dimension();
        ndassert!(
            a.ndim() == chunk.ndim(),
            concat!(
                "Chunk dimension {} does not match array dimension {} ",
                "(with array of shape {:?})"
            ),
            chunk.ndim(),
            a.ndim(),
            a.shape()
        );
        for i in 0..a.ndim() {
            a.dim[i] /= chunk[i];
        }
        let inner_strides = a.raw_strides();
        a.strides *= &chunk;

        ExactChunks {
            base: a,
            chunk,
            inner_strides,
        }
    }
}

impl<'a, A, D> IntoIterator for ExactChunks<'a, A, D>
where
    D: Dimension,
    A: 'a,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = ExactChunksIter<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter {
        unsafe {
            ExactChunksIter::new(
                self.base.ptr.as_ptr(),
                self.base.dim,
                self.base.strides,
                BIItemArrayViewInner::new(self.chunk, self.inner_strides),
            )
        }
    }
}

impl_ndproducer! {
    ['a, A, D: Dimension]
    [Clone => ]
    ExactChunksMut {
        base,
        chunk,
        inner_strides,
    }
    ExactChunksMut<'a, A, D> {
        type Item = ArrayViewMut<'a, A, D>;
        type Dim = D;

        unsafe fn item(&self, ptr) {
            ArrayViewMut::new_(ptr,
                               self.chunk.clone(),
                               self.inner_strides.clone())
        }
    }
}

/// Exact chunks producer and iterable.
///
/// See [`.exact_chunks_mut()`](ArrayBase::exact_chunks_mut)
/// for more information.
//#[derive(Debug)]
pub struct ExactChunksMut<'a, A, D> {
    base: BaseProducerMut<'a, A, D>,
    chunk: D,
    inner_strides: D,
}

impl<'a, A, D: Dimension> ExactChunksMut<'a, A, D> {
    /// Creates a new exact chunks producer.
    ///
    /// **Panics** if any chunk dimension is zero
    pub(crate) fn new<E>(mut a: ArrayViewMut<'a, A, D>, chunk: E) -> Self
    where
        E: IntoDimension<Dim = D>,
    {
        let chunk = chunk.into_dimension();
        ndassert!(
            a.ndim() == chunk.ndim(),
            concat!(
                "Chunk dimension {} does not match array dimension {} ",
                "(with array of shape {:?})"
            ),
            chunk.ndim(),
            a.ndim(),
            a.shape()
        );
        for i in 0..a.ndim() {
            a.dim[i] /= chunk[i];
        }
        let inner_strides = a.raw_strides();
        a.strides *= &chunk;

        ExactChunksMut {
            base: a,
            chunk,
            inner_strides,
        }
    }
}

impl<'a, A, D> IntoIterator for ExactChunksMut<'a, A, D>
where
    D: Dimension,
    A: 'a,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = ExactChunksIterMut<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter {
        unsafe {
            ExactChunksIterMut::new(
                self.base.ptr.as_ptr(),
                self.base.dim,
                self.base.strides,
                BIItemArrayViewInner::new(self.chunk, self.inner_strides),
            )
        }
    }
}
