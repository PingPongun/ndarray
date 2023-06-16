use rayon::iter::plumbing::bridge;
use rayon::iter::plumbing::bridge_unindexed;
use rayon::iter::plumbing::Folder;
use rayon::iter::plumbing::Producer;
use rayon::iter::plumbing::ProducerCallback;
use rayon::iter::plumbing::UnindexedProducer;
use rayon::iter::plumbing::{Consumer, UnindexedConsumer};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelIterator;

use crate::iterators::BIItemArrayView;
use crate::iterators::BIItemArrayViewMut;
use crate::iterators::BIItemRef;
use crate::iterators::BIItemRefMut;
use crate::iterators::BIItemVariableArrayView;
use crate::iterators::BIItemVariableArrayViewMut;
use crate::split_at::SplitPreference;
use crate::BaseIter;
use crate::Dimension;
use crate::{ArrayView, ArrayViewMut};

/// Parallel iterator wrapper.
#[derive(Copy, Clone, Debug)]
pub struct Parallel<I> {
    iter: I,
    min_len: usize,
}

const DEFAULT_MIN_LEN: usize = 1;

/// Parallel producer wrapper.
#[derive(Copy, Clone, Debug)]
struct ParallelProducer<I>(I, usize);

macro_rules! BI_par_iter_wrapper {
    // thread_bounds are either Sync or Send + Sync
    ([$($generics:tt)*], $item:ty,$idx:expr, [$($thread_bounds:tt)*], [$($add_bounds:tt)*]) => {
    /// Requires crate feature `rayon`.
    impl<'a, A, D,$($generics)*> IntoParallelIterator for BaseIter<A, D, $idx, false, $item>
    where D: Dimension,
        A: $($thread_bounds)*,
        $($add_bounds)*
    {
        type Item = <Self as Iterator>::Item;
        type Iter = Parallel<Self>;
        fn into_par_iter(self) -> Self::Iter {
            Parallel {
                iter: self,
                min_len: DEFAULT_MIN_LEN,
            }
        }
    }
    impl<'a, A, D,$($generics)*> ParallelIterator for Parallel<BaseIter<A, D, $idx, false, $item>>
    where D: Dimension,
        A: $($thread_bounds)*,
        $($add_bounds)*
    {
        type Item = <BaseIter<A, D, $idx, false, $item> as Iterator>::Item;
        fn drive_unindexed<C>(self, consumer: C) -> C::Result
            where C: UnindexedConsumer<Self::Item>
        {
            bridge(self, consumer)
        }

        fn opt_len(&self) -> Option<usize> {
            Some(self.iter.len())
        }
    }


    impl<'a, A, D,$($generics)*> IndexedParallelIterator for Parallel<BaseIter<A, D, $idx, false, $item>>
    where D: Dimension,
        A: $($thread_bounds)*,
        $($add_bounds)*
    {
        fn with_producer<Cb>(self, callback: Cb) -> Cb::Output
            where Cb: ProducerCallback<Self::Item>
        {
            callback.callback(ParallelProducer(self.iter, self.min_len))
        }

        fn len(&self) -> usize {
            ExactSizeIterator::len(&self.iter)
        }

        fn drive<C>(self, consumer: C) -> C::Result
            where C: Consumer<Self::Item>
        {
            bridge(self, consumer)
        }
    }

        impl<'a, A, D,$($generics)*> IntoIterator for ParallelProducer<BaseIter<A, D, $idx, false, $item>>
        where D: Dimension,
            A: $($thread_bounds)*,
            $($add_bounds)*
    {
        type IntoIter = BaseIter<A, D, $idx, false, $item>;
        type Item = <Self::IntoIter as Iterator>::Item;

        fn into_iter(self) -> Self::IntoIter {
            self.0
        }
    }

    // This is the real magic, I guess
    impl<'a, A, D,$($generics)*> Producer for ParallelProducer<BaseIter<A, D, $idx, false, $item>>
    where D: Dimension,
        A: $($thread_bounds)*,
        $($add_bounds)*
    {
        type IntoIter = BaseIter<A, D, $idx, false, $item>;
        type Item = <Self::IntoIter as Iterator>::Item;

        fn into_iter(self) -> Self::IntoIter {
            self.0
        }

        fn split_at(self, i: usize) -> (Self, Self) {
            let (a, b) = self.0.split_at(i);
            (ParallelProducer(a, self.1), ParallelProducer(b, self.1))
        }
    }

    };
    ([$($generics:tt)*], $item:ty, [$($thread_bounds:tt)*]) => {
        BI_par_iter_wrapper!([$($generics)*], $item, false, [$($thread_bounds)*], []);
        BI_par_iter_wrapper!([$($generics)*], $item, true, [$($thread_bounds)*], [D::Pattern: Send]);
    }
}

BI_par_iter_wrapper!([], BIItemRef<'a, A>, [Sync]);
BI_par_iter_wrapper!([], BIItemRefMut<'a, A>, [Sync + Send]);
BI_par_iter_wrapper!(
    [DI: Dimension],
    BIItemArrayView<'a, A, DI>,
    [Sync]
);
BI_par_iter_wrapper!(
    [DI: Dimension],
    BIItemArrayViewMut<'a, A, DI>,
    [Sync + Send]
);
BI_par_iter_wrapper!(
    [DI: Dimension],
    BIItemVariableArrayView<'a, A, DI>,
    [Sync]
);
BI_par_iter_wrapper!(
    [DI: Dimension],
    BIItemVariableArrayViewMut<'a, A, DI>,
    [Sync + Send]
);

macro_rules! par_iter_view_wrapper {
    // thread_bounds are either Sync or Send + Sync
    ($view_name:ident, [$($thread_bounds:tt)*]) => {
    /// Requires crate feature `rayon`.
    impl<'a, A, D> IntoParallelIterator for $view_name<'a, A, D>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        type Item = <Self as IntoIterator>::Item;
        type Iter = Parallel<Self>;
        fn into_par_iter(self) -> Self::Iter {
            Parallel {
                iter: self,
                min_len: DEFAULT_MIN_LEN,
            }
        }
    }

    impl<'a, A, D> ParallelIterator for Parallel<$view_name<'a, A, D>>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        type Item = <$view_name<'a, A, D> as IntoIterator>::Item;
        fn drive_unindexed<C>(self, consumer: C) -> C::Result
            where C: UnindexedConsumer<Self::Item>
        {
            bridge_unindexed(ParallelProducer(self.iter, self.min_len), consumer)
        }

        fn opt_len(&self) -> Option<usize> {
            None
        }
    }

    impl<'a, A, D> Parallel<$view_name<'a, A, D>>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        /// Sets the minimum number of elements desired to process in each job. This will not be
        /// split any smaller than this length, but of course a producer could already be smaller
        /// to begin with.
        ///
        /// ***Panics*** if `min_len` is zero.
        pub fn with_min_len(self, min_len: usize) -> Self {
            assert_ne!(min_len, 0, "Minimum number of elements must at least be one to avoid splitting off empty tasks.");

            Self {
                min_len,
                ..self
            }
        }
    }

    impl<'a, A, D> UnindexedProducer for ParallelProducer<$view_name<'a, A, D>>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        type Item = <$view_name<'a, A, D> as IntoIterator>::Item;
        fn split(self) -> (Self, Option<Self>) {
            if self.0.len() <= self.1 {
                return (self, None)
            }
            let array = self.0;
            let max_axis = array.max_stride_axis();
            let mid = array.len_of(max_axis) / 2;
            let (a, b) = array.split_at(max_axis, mid);
            (ParallelProducer(a, self.1), Some(ParallelProducer(b, self.1)))
        }

        fn fold_with<F>(self, folder: F) -> F
            where F: Folder<Self::Item>,
        {
            Zip::from(self.0).fold_while(folder, |mut folder, elt| {
                folder = folder.consume(elt);
                if folder.full() {
                    FoldWhile::Done(folder)
                } else {
                    FoldWhile::Continue(folder)
                }
            }).into_inner()
        }
    }

    impl<'a, A, D> IntoIterator for ParallelProducer<$view_name<'a, A, D>>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        type Item = <$view_name<'a, A, D> as IntoIterator>::Item;
        type IntoIter = <$view_name<'a, A, D> as IntoIterator>::IntoIter;
        fn into_iter(self) -> Self::IntoIter {
            self.0.into_iter()
        }
    }

    }
}

par_iter_view_wrapper!(ArrayView, [Sync]);
par_iter_view_wrapper!(ArrayViewMut, [Sync + Send]);

use crate::{FoldWhile, NdProducer, Zip};

macro_rules! zip_impl {
    ($([$($p:ident)*],)+) => {
        $(
        /// Requires crate feature `rayon`.
        #[allow(non_snake_case)]
        impl<D, $($p),*> IntoParallelIterator for Zip<($($p,)*), D>
            where $($p::Item : Send , )*
                  $($p : Send , )*
                  D: Dimension,
                  $($p: NdProducer<Dim=D> ,)*
        {
            type Item = ($($p::Item ,)*);
            type Iter = Parallel<Self>;
            fn into_par_iter(self) -> Self::Iter {
                Parallel {
                    iter: self,
                    min_len: DEFAULT_MIN_LEN,
                }
            }
        }

        #[allow(non_snake_case)]
        impl<D, $($p),*> ParallelIterator for Parallel<Zip<($($p,)*), D>>
            where $($p::Item : Send , )*
                  $($p : Send , )*
                  D: Dimension,
                  $($p: NdProducer<Dim=D> ,)*
        {
            type Item = ($($p::Item ,)*);

            fn drive_unindexed<Cons>(self, consumer: Cons) -> Cons::Result
                where Cons: UnindexedConsumer<Self::Item>
            {
                bridge_unindexed(ParallelProducer(self.iter, self.min_len), consumer)
            }

            fn opt_len(&self) -> Option<usize> {
                None
            }
        }

        #[allow(non_snake_case)]
        impl<D, $($p),*> UnindexedProducer for ParallelProducer<Zip<($($p,)*), D>>
            where $($p : Send , )*
                  $($p::Item : Send , )*
                  D: Dimension,
                  $($p: NdProducer<Dim=D> ,)*
        {
            type Item = ($($p::Item ,)*);

            fn split(self) -> (Self, Option<Self>) {
                if self.0.size() <= self.1 {
                    return (self, None)
                }
                let (a, b) = self.0.split();
                (ParallelProducer(a, self.1), Some(ParallelProducer(b, self.1)))
            }

            fn fold_with<Fold>(self, folder: Fold) -> Fold
                where Fold: Folder<Self::Item>,
            {
                self.0.fold_while(folder, |mut folder, $($p),*| {
                    folder = folder.consume(($($p ,)*));
                    if folder.full() {
                        FoldWhile::Done(folder)
                    } else {
                        FoldWhile::Continue(folder)
                    }
                }).into_inner()
            }
        }
        )+
    }
}

zip_impl! {
    [P1],
    [P1 P2],
    [P1 P2 P3],
    [P1 P2 P3 P4],
    [P1 P2 P3 P4 P5],
    [P1 P2 P3 P4 P5 P6],
}

impl<D, Parts> Parallel<Zip<Parts, D>>
where
    D: Dimension,
{
    /// Sets the minimum number of elements desired to process in each job. This will not be
    /// split any smaller than this length, but of course a producer could already be smaller
    /// to begin with.
    ///
    /// ***Panics*** if `min_len` is zero.
    pub fn with_min_len(self, min_len: usize) -> Self {
        assert_ne!(min_len, 0, "Minimum number of elements must at least be one to avoid splitting off empty tasks.");

        Self {
            min_len,
            ..self
        }
    }
}

/// A parallel iterator (unindexed) that produces the splits of the array
/// or producer `P`.
pub(crate) struct ParallelSplits<P> {
    pub(crate) iter: P,
    pub(crate) max_splits: usize,
}

impl<P> ParallelIterator for ParallelSplits<P>
    where P: SplitPreference + Send,
{
    type Item = P;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where C: UnindexedConsumer<Self::Item>
    {
        bridge_unindexed(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        None
    }
}

impl<P> UnindexedProducer for ParallelSplits<P>
    where P: SplitPreference + Send,
{
    type Item = P;

    fn split(self) -> (Self, Option<Self>) {
        if self.max_splits == 0 || !self.iter.can_split() {
            return (self, None)
        }
        let (a, b) = self.iter.split();
        (ParallelSplits {
            iter: a,
            max_splits: self.max_splits - 1,
        },
        Some(ParallelSplits {
            iter: b,
            max_splits: self.max_splits - 1,
        }))
    }

    fn fold_with<Fold>(self, folder: Fold) -> Fold
        where Fold: Folder<Self::Item>,
    {
        folder.consume(self.iter)
    }
}
