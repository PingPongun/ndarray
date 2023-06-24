// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[macro_use]
mod macros;
mod chunks;
mod into_iter;
pub mod iter;
mod lanes;
mod windows;

use alloc::vec::Vec;
use core::hint::unreachable_unchecked;
use std::marker::PhantomData;
use std::ptr;

use crate::Ix1;

pub use self::chunks::{ExactChunks, ExactChunksMut};
pub use self::into_iter::IntoIter;
pub use self::lanes::{Lanes, LanesMut};
pub use self::windows::Windows;
use super::{ArrayView, ArrayViewMut, Axis, NdProducer};
use super::{Dimension, Ixs};
use core::fmt::Debug;
use std::slice::{self};

pub struct BaseIter0d<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> {
    ptr: *mut A,
    elems_left: usize,
    inner: IdxA::Inner,
    _item: PhantomData<IdxA>,
}

pub struct BaseIter1d<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> {
    ptr: *mut A,
    dim: D,
    strides: D,
    end: D,
    index: D,
    standard_layout: bool,
    inner: IdxA::Inner,
    _item: PhantomData<IdxA>,
}

pub struct BaseIterNd<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> {
    ptr: *mut A,
    dim: D,
    strides: D,
    end: D,
    elems_left: usize,
    index: D,
    elems_left_row: [usize; 2],
    elems_left_row_back_idx: usize,
    offset_front: isize,
    offset_back: isize,
    inner: IdxA::Inner,
    _item: PhantomData<IdxA>,
}

/// BaseIterNdSINGLE_ENDED_FOLDING- BaseIterNd optimized for single ended use (does not implement DoubleEndedIterator & ParallelIter)
/// intended mostly for simple use cases as array.iter().fold(...), array.iter.sum(), ..
/// iterating with next() may be less performant(in some cases) than with BaseIterNd, but creation time is shorter (especially important for small arrays & IxDyn)
pub struct BaseIterNdSEF<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> {
    ptr: *mut A,
    dim: D,
    strides: D,
    elems_left: usize,
    elems_left_row: [usize; 1],
    index: D,
    offset_front: isize,
    inner: IdxA::Inner,
    _item: PhantomData<IdxA>,
}

/// Base for iterators over all axes.
///
/// Iterator element type is `*mut A`.
/// index and end values are only valid indices when elements_left >= 1
pub enum BaseIter<A, D: Dimension, const IDX: bool, const SEF: bool, IdxA: BIItemT<A, D, IDX>> {
    D0(BaseIter0d<A, D, IDX, IdxA>),
    D1(BaseIter1d<A, D, IDX, IdxA>),
    Dn(BaseIterNd<A, D, IDX, IdxA>),
    DnSEF(BaseIterNdSEF<A, D, IDX, IdxA>),
}

#[macro_use]
pub(crate) mod _macros {
    macro_rules! eitherBI {
        ($bi:expr, $SEF:expr, $SEF_expr:expr, $inner:pat => $result:expr) => {
            match D::NDIM {
                Some(0) => {
                    unwrapBI!($bi,D0,$inner=>$result)
                }
                Some(1) => {
                    unwrapBI!($bi,D1,$inner=>$result)
                }
                _ => {
                    if $SEF {
                        $SEF_expr
                    } else {
                        unwrapBI!($bi,Dn,$inner=>$result)
                    }
                }
            }
        };
        ($bi:expr, $inner:pat => $result:expr) => {
            eitherBI!($bi, SEF, unwrapBI!($bi,DnSEF,$inner=>$result), $inner=>$result)
        };
        (nSEF,$bi:expr, $inner:pat => $result:expr) => {
            eitherBI!($bi, false, unsafe{unreachable_unchecked()}, $inner=>$result)
        };
    }

    macro_rules! eitherBIwrapped {
        ($bi:expr, $SEF:expr, $SEF_expr:expr, $inner:pat => $result:expr) => {
            match D::NDIM {
                Some(0) => {
                    unwrapBI!($bi,D0,$inner=>BaseIter::D0($result))
                }
                Some(1) => {
                    unwrapBI!($bi,D1,$inner=>BaseIter::D1($result))
                }
                _ => {
                    if $SEF {
                        $SEF_expr
                    } else {
                        unwrapBI!($bi,Dn,$inner=>BaseIter::Dn($result))
                    }
                }
            }
        };
        ($bi:expr, $inner:pat => $result:expr) => {
            eitherBIwrapped!($bi, SEF, unwrapBI!($bi,DnSEF,$inner=>BaseIter::DnSEF($result)), $inner=>$result)
        };
        (nSEF,$bi:expr, $inner:pat => $result:expr) => {
            eitherBIwrapped!($bi, false,  unsafe{unreachable_unchecked()}, $inner=>$result)
        };
    }
    macro_rules! ifIdx {
        ( $func:expr , $jump:expr) => {
            if IdxA::REQUIRES_IDX {
                let ret = $func;
                $jump;
                ret
            } else {
                $func
            }
        };
    }
    macro_rules! IdxA {
        ($inner:expr,$idx:expr,$ptr:expr) => {
            if IdxA::REQUIRES_IDX {
                IdxA::item_w_idx(&$inner, $idx, $ptr)
            } else {
                IdxA::item(&$inner, $ptr)
            }
        };
    }

    macro_rules! BaseIterNdFoldOuterLoop {
        ($_self:ident, $idx:ident, $idx_jump_h:ident, $idx_def:expr, $offset:ident, $offset_func:ident, $inner_loop:expr) => {
            loop {
                $_self.elems_left -= $_self.elems_left_row[0];
                $inner_loop;
                if $_self.elems_left > $_self.dim.last_elem() {
                    $_self.elems_left_row[0] = $_self.dim.last_elem();
                } else if $_self.elems_left == 0 {
                    break;
                } else {
                    $_self.elems_left_row[0] = $_self.elems_left;
                }
                $_self.dim.$idx_jump_h(&mut $_self.$idx);
                $_self.$offset = D::$offset_func(&$_self.$idx, &$_self.strides);
                if IdxA::REQUIRES_IDX {
                    $_self.$idx.set_last_elem($idx_def);
                }
            }
        };
    }

    macro_rules! BaseIterNdFold {
        ($_self:ident,$init:ident,$g:ident,$stride:ident,$idx:ident, $idx_step:ident,$idx_jump_h:ident,$idx_def:expr,$pre_fold:expr,$fold:ident, $offset:ident, $offset_func:ident) => {
            let mut accum = $init;
            $_self.elems_left += $_self.elems_left_row[0];
            $pre_fold;
            if $_self.strides.last_elem() == 1 {
                BaseIterNdFoldOuterLoop! {
                    $_self, $idx, $idx_jump_h, $idx_def, $offset, $offset_func,
                    {
                        accum = unsafe {
                            core::slice::from_raw_parts_mut(
                                $_self.ptr.offset($_self.$offset),
                                $_self.elems_left_row[0],
                            )
                        }
                        .iter_mut()
                        .$fold(accum, |acc, ptr| {
                            let new_acc=$g(acc, IdxA!($_self.inner, $_self.$idx.clone(), ptr));
                            if IdxA::REQUIRES_IDX {
                                $_self.$idx.$idx_step(1);
                            }
                            new_acc
                        });
                    }
                }
            } else {
                BaseIterNdFoldOuterLoop! {
                    $_self, $idx, $idx_jump_h, $idx_def, $offset, $offset_func,
                    {
                        while 0 != $_self.elems_left_row[0] {
                            $_self.elems_left_row[0] -= 1;
                            unsafe {
                                accum = $g(
                                    accum,
                                    IdxA!(
                                        $_self.inner,
                                        $_self.$idx.clone(),
                                        $_self.ptr.offset($_self.$offset)
                                    ),
                                );
                            }
                            $_self.$offset += $stride;
                            if IdxA::REQUIRES_IDX {
                                $_self.$idx.$idx_step(1);
                            }
                        }
                    }
                }
            }
            return accum;
        };
        (nSEF, $_self:ident,$init:ident,$g:ident,$stride:ident,$idx:ident, $idx_step:ident,$idx_jump_h:ident,$idx_def:expr,$pre_fold:expr,$fold:ident, $offset:ident, $offset_func:ident) => {
            BaseIterNdFold!(
                $_self,
                $init,
                $g,
                $stride,
                $idx,
                $idx_step,
                $idx_jump_h,
                $idx_def,
                {
                    $_self.elems_left += $_self.elems_left_row[1];
                    $_self.elems_left_row_back_idx = 0;
                    $_self.elems_left_row[1] = 0;
                    $pre_fold
                },
                $fold,
                $offset,
                $offset_func
            )
        };
    }
    macro_rules! BaseIterNdNext {
        ($_self:ident,$stride:ident,$idx:ident, $idx_step:ident,$idx_jump_h:ident,$idx_def:expr, $elems_idx:expr,$forward:ident,$offset:ident, $offset_func:ident,$elems_left_0_expr:expr) => {
            if $_self.elems_left_row[$elems_idx] <= 1 {
                //last element in row
                if $_self.elems_left == 0 {
                    $elems_left_0_expr
                } else {
                    //last element in each row
                    let index = $_self.$idx.clone();
                    let ret = unsafe { $_self.ptr.offset($_self.$offset) };
                    $_self.elems_left -= $_self.dim.last_elem();
                    $_self.elems_left_row[$elems_idx] = $_self.dim.last_elem();
                    $_self.dim.$idx_jump_h(&mut $_self.$idx); //switch to new row
                    if IdxA::REQUIRES_IDX {
                        $_self.$idx.set_last_elem($idx_def);
                    }
                    $_self.$offset = D::$offset_func(&$_self.$idx, &$_self.strides);
                    Some(IdxA!($_self.inner, index, ret))
                }
            } else {
                //normal(not last in row) element
                $_self.elems_left_row[$elems_idx] -= 1;
                let index = $_self.$idx.clone();
                let ret = unsafe { $_self.ptr.offset($_self.$offset) };
                $_self.$offset += $stride;
                if IdxA::REQUIRES_IDX {
                    $_self.$idx.$idx_step(1);
                }
                Some(IdxA!($_self.inner, index, ret))
            }
        };
        (SEF,$_self:ident,$stride:ident,$idx:ident, $idx_step:ident,$idx_jump_h:ident,$idx_def:expr, $elems_idx:expr,$forward:ident,$offset:ident, $offset_func:ident) => {
            BaseIterNdNext!(
                $_self,
                $stride,
                $idx,
                $idx_step,
                $idx_jump_h,
                $idx_def,
                $elems_idx,
                $forward,
                $offset,
                $offset_func,
                {
                    //we are already in last row
                    if $_self.elems_left_row[0] == 0 {
                        None
                    } else {
                        //last elem of last row
                        $_self.elems_left_row[0] = 0;
                        let index = $_self.$idx.clone();
                        let ret = unsafe { $_self.ptr.offset($_self.$offset) };
                        Some(IdxA!($_self.inner, index, ret))
                    }
                }
            )
        };
        ($_self:ident,$stride:ident,$idx:ident, $idx_step:ident,$idx_jump_h:ident,$idx_def:expr, $elems_idx:expr,$forward:ident,$offset:ident, $offset_func:ident) => {
            BaseIterNdNext!(
                $_self,
                $stride,
                $idx,
                $idx_step,
                $idx_jump_h,
                $idx_def,
                $elems_idx,
                $forward,
                $offset,
                $offset_func,
                {
                    //elems_left is multiple of self.dim.last_elem()
                    //there are no "untouched" rows
                    if $_self.elems_left_row_back_idx == 0 {
                        //we are already in last row
                        if $_self.elems_left_row[0] == 0 {
                            None
                        } else {
                            //last elem of last row
                            $_self.elems_left_row[0] = 0;
                            let index = $_self.$idx.clone();
                            let ret = unsafe { $_self.ptr.offset($_self.$offset) };
                            Some(IdxA!($_self.inner, index, ret))
                        }
                    } else {
                        //switching to last row
                        let index = $_self.$idx.clone();
                        let ret = unsafe { $_self.ptr.offset($_self.$offset) };
                        if $forward {
                            $_self.elems_left_row[0] = $_self.elems_left_row[1];
                            $_self.elems_left_row[1] = 0;
                        }
                        $_self.elems_left_row_back_idx = 0;
                        $_self.dim.$idx_jump_h(&mut $_self.$idx); //switch to new row
                        if IdxA::REQUIRES_IDX {
                            $_self.$idx.set_last_elem($idx_def);
                        }
                        $_self.$offset = D::$offset_func(&$_self.$idx, &$_self.strides);
                        Some(IdxA!($_self.inner, index, ret))
                    }
                }
            )
        };
    }
    macro_rules! impl_BIItem {
        ( $name:ident, [$($generics_ty:tt)*], [$($generics:tt)*], [$($generics_constr:tt)*], $ret:ty,$inner_ty:ty, $inner:ident,$idx:ident,$req_idx:expr,$idx_op:expr,$n_idx_item:expr, $pat:pat => $func:expr) => {

            pub struct $name< $($generics_ty)*, $($generics)*>(PhantomData<$ret>);

            impl<'a, A, D: Dimension, $($generics_constr)*> BIItemT<A, D, true> for $name<$($generics_ty)*, $($generics)*> {
                type BIItem = (D::Pattern, $ret);
                type Inner = $inner_ty;
                const REQUIRES_IDX: bool = true;
                const NAME: &'static str = concat!("Indexed ", stringify!($name));

                #[inline(always)]
                fn item_w_idx(
                    $inner: &Self::Inner,
                    $idx: D,
                    $pat: *mut A,
                ) -> Self::BIItem {
                    ($idx_op, $func)
                }
            }
            impl<'a, A, D: Dimension, $($generics_constr)*> BIItemT<A, D, false> for $name<$($generics_ty)*, $($generics)*> {
                type BIItem = $ret;
                type Inner = $inner_ty;
                const REQUIRES_IDX: bool = $req_idx;
                const NAME: &'static str = concat!("Unindexed ", stringify!($name));

                #[inline(always)]
                fn item($inner: &Self::Inner,
                    $pat: *mut A) -> Self::BIItem {
                    $n_idx_item
                }
                #[inline(always)]
                fn item_w_idx($inner: &Self::Inner,
                    $idx: D, $pat: *mut A) -> Self::BIItem {
                    $func
                }
            }
        };
        ( $name:ident, [$($generics_ty:tt)*], [$($generics:tt)*], [$($generics_constr:tt)*], $ret:ty,$inner_ty:ty, $inner:ident,$idx:ident, $pat:pat => $func:expr) => {
            impl_BIItem!( $name, [$($generics_ty)*], [$($generics)*], [$($generics_constr)*], $ret, $inner_ty, $inner, $idx, true, $idx.clone().into_pattern(), unsafe{unreachable_unchecked()}, $pat => $func);
        };
        ( $name:ident, [$($generics_ty:tt)*], [$($generics:tt)*], [$($generics_constr:tt)*], $ret:ty,$inner_ty:ty, $inner:ident, $pat:pat => $func:expr) => {
            impl_BIItem!( $name, [$($generics_ty)*], [$($generics)*], [$($generics_constr)*], $ret, $inner_ty, $inner, _idx, false, _idx.into_pattern(), $func, $pat => $func);
        };
        ( $name:ident, [$($generics_ty:tt)*], $ret:ty , $pat:pat => $func:expr) => {
            impl_BIItem!( $name, [$($generics_ty)*], [], [], $ret, (), _inner, $pat => $func);
        };
    }
}

pub trait BIItemT<A, D: Dimension, const IDX: bool> {
    type BIItem;
    type Inner: Clone + Debug;
    const REQUIRES_IDX: bool;
    const NAME: &'static str;
    #[inline(always)]
    fn item(_inner: &Self::Inner, _val: *mut A) -> Self::BIItem {
        unsafe {
            unreachable_unchecked();
        }
    }
    #[inline(always)]
    fn item_w_idx(_inner: &Self::Inner, _idx: D, _val: *mut A) -> Self::BIItem {
        unsafe {
            unreachable_unchecked();
        }
    }
}
#[derive(Debug, Clone)]
pub struct BIItemArrayViewInner<DI: Dimension> {
    pub dim: DI,
    pub strides: DI,
}
impl<DI: Dimension> BIItemArrayViewInner<DI> {
    pub fn new(dim: DI, strides: DI) -> Self {
        Self { dim, strides }
    }
}
#[derive(Debug, Clone)]
pub struct BIItemVariableArrayViewInner<DI: Dimension> {
    pub dim: DI,
    pub strides: DI,
    pub remainder_index: usize,
    pub remainder_dim: DI,
}

impl<DI: Dimension> BIItemVariableArrayViewInner<DI> {
    pub fn new(dim: DI, strides: DI, remainder_index: usize, remainder_dim: DI) -> Self {
        Self {
            dim,
            strides,
            remainder_index,
            remainder_dim,
        }
    }
}
impl_BIItem!(BIItemPtr, [A], *mut A,ptr =>ptr);
impl_BIItem!(BIItemRef, ['a, A], &'a A,ptr => unsafe{&*ptr});
impl_BIItem!(BIItemRefMut, ['a, A], &'a mut A,ptr => unsafe{&mut *ptr});
impl_BIItem!(BIItemArrayView, ['a, A], [DI], [DI: Dimension], ArrayView<'a,A,DI>, BIItemArrayViewInner<DI>, inner,
    ptr => unsafe{ArrayView::new_(ptr,inner.dim.clone(), inner.strides.clone() )});
impl_BIItem!(BIItemArrayViewMut, ['a, A], [DI], [DI: Dimension], ArrayViewMut<'a,A,DI>, BIItemArrayViewInner<DI>, inner,
    ptr => unsafe{ArrayViewMut::new_(ptr,inner.dim.clone(), inner.strides.clone() )});
impl_BIItem!(BIItemVariableArrayView, ['a, A], [DI], [DI: Dimension], ArrayView<'a,A,DI>, BIItemVariableArrayViewInner<DI>, _inner,idx,
_ptr => {
    if D::NDIM == Some(1) {
        if idx[0] == _inner.remainder_index {
            unsafe { ArrayView::new_(_ptr, _inner.remainder_dim.clone(), _inner.strides.clone()) }
        } else {
            unsafe { ArrayView::new_(_ptr, _inner.dim.clone(), _inner.strides.clone()) }
        }
    } else {
        unimplemented!();
    }
});
impl_BIItem!(BIItemVariableArrayViewMut, ['a, A], [DI], [DI: Dimension], ArrayViewMut<'a,A,DI>, BIItemVariableArrayViewInner<DI>, _inner,idx,
_ptr => {
    if D::NDIM == Some(1) {
        if idx[0] == _inner.remainder_index {
            unsafe { ArrayViewMut::new_(_ptr, _inner.remainder_dim.clone(), _inner.strides.clone()) }
        } else {
            unsafe { ArrayViewMut::new_(_ptr, _inner.dim.clone(), _inner.strides.clone()) }
        }
    } else {
        unimplemented!();
    }
});

//=================================================================================================
mod base_iter_0d {
    use super::*;
    impl<A: Debug, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> Debug
        for BaseIter0d<A, D, IDX, IdxA>
    {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("BaseIter0d")
                .field("ptr", &self.ptr)
                .field("elems_left", &self.elems_left)
                .field("inner", &self.inner)
                .finish()
        }
    }

    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> BaseIter0d<A, D, IDX, IdxA> {
        /// Creating a Baseiter is unsafe because shape and stride parameters need
        /// to be correct to avoid performing an unsafe pointer offset while
        /// iterating.
        #[inline(always)]
        pub unsafe fn new(ptr: *mut A, inner: IdxA::Inner) -> BaseIter0d<A, D, IDX, IdxA> {
            BaseIter0d {
                ptr,
                elems_left: 1,
                inner,
                _item: PhantomData,
            }
        }

        /// Splits the iterator at `index`, yielding two disjoint iterators.
        ///
        /// `index` is relative to the current state of the iterator (which is not
        /// necessarily the start of the axis).
        ///
        /// **Panics** if `index` is strictly greater than the iterator's remaining
        /// length.
        #[inline(always)]
        pub(crate) fn split_at(self, _index: usize) -> (Self, Self) {
            let mut right = self.clone();
            right.elems_left = 0;
            (self, right)
        }
        #[inline]
        pub(crate) fn consume(&mut self) {
            self.elems_left = 0;
        }
    }

    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> Iterator
        for BaseIter0d<A, D, IDX, IdxA>
    {
        type Item = IdxA::BIItem;

        #[inline(always)]
        fn next(&mut self) -> Option<Self::Item> {
            if self.elems_left == 1 {
                self.elems_left = 0;
                Some(IdxA!(self.inner, D::default(), self.ptr))
            } else {
                None
            }
        }

        #[inline(always)]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = self.len();
            (len, Some(len))
        }
    }

    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> DoubleEndedIterator
        for BaseIter0d<A, D, IDX, IdxA>
    {
        #[inline(always)]
        fn next_back(&mut self) -> Option<Self::Item> {
            self.next()
        }
    }
    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> ExactSizeIterator
        for BaseIter0d<A, D, IDX, IdxA>
    {
        #[inline(always)]
        fn len(&self) -> usize {
            self.elems_left
        }
    }

    clone_bounds!(
        [A, D: Clone+Dimension,const IDX:bool, IdxA: BIItemT<A, D, IDX>]
        BaseIter0d[A, D, IDX, IdxA] {
            @copy {
                ptr,
            }
            elems_left,
            inner,
            _item,
        }
    );
}
//=================================================================================================
mod base_iter_1d {
    use super::*;
    impl<A: Debug, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> Debug
        for BaseIter1d<A, D, IDX, IdxA>
    {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("BaseIter1d")
                .field("ptr", &self.ptr)
                .field("dim", &self.dim)
                .field("index", &self.index)
                .field("end", &self.end)
                .field("strides", &self.strides)
                .field("standard_layout", &self.standard_layout)
                .field("inner", &self.inner)
                .finish()
        }
    }

    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> BaseIter1d<A, D, IDX, IdxA> {
        /// Creating a Baseiter is unsafe because shape and stride parameters need
        /// to be correct to avoid performing an unsafe pointer offset while
        /// iterating.
        #[inline(always)]
        pub unsafe fn new(ptr: *mut A, len: D, strides: D, inner: IdxA::Inner) -> Self {
            let end = len.clone();
            let standard_layout = len.is_layout_c_unchecked(&strides);
            BaseIter1d {
                ptr,
                index: D::zeros(len.ndim()),
                dim: len,
                strides,
                end,
                standard_layout,
                inner,
                _item: PhantomData,
            }
        }

        /// Splits the iterator at `index`, yielding two disjoint iterators.
        ///
        /// `index` is relative to the current state of the iterator (which is not
        /// necessarily the start of the axis).
        ///
        /// **Panics** if `index` is strictly greater than the iterator's remaining
        /// length.
        #[inline(always)]
        pub(crate) fn split_at(self, index: usize) -> (Self, Self) {
            assert!(index <= self.len());
            let mut mid = self.index.clone();
            self.dim.jump_index_by_unchecked(&mut mid, index);
            let left = BaseIter1d {
                index: self.index,
                dim: self.dim.clone(),
                strides: self.strides.clone(),
                ptr: self.ptr,
                end: mid.clone(),
                standard_layout: self.standard_layout,
                inner: self.inner.clone(),
                _item: self._item,
            };
            let right = BaseIter1d {
                index: mid,
                dim: self.dim,
                strides: self.strides,
                ptr: self.ptr,
                end: self.end,
                standard_layout: self.standard_layout,
                inner: self.inner,
                _item: self._item,
            };
            (left, right)
        }
        #[inline]
        pub(crate) fn consume(&mut self) {
            self.index.set_last_elem(self.end.last_elem());
        }
    }

    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> Iterator
        for BaseIter1d<A, D, IDX, IdxA>
    {
        type Item = IdxA::BIItem;

        #[inline(always)]
        fn next(&mut self) -> Option<Self::Item> {
            if self.index.last_elem() == self.end.last_elem() {
                None
            } else {
                let ret = unsafe {
                    self.ptr
                        .offset(D::stride_offset(&self.index, &self.strides))
                };
                let index = self.index.clone();
                self.index.last_wrapping_add(1);
                Some(IdxA!(self.inner, index, ret))
            }
        }

        #[inline(always)]
        fn nth(&mut self, count: usize) -> Option<Self::Item> {
            if self.len() <= count {
                self.index.set_last_elem(self.end.last_elem());
                None
            } else {
                self.index.last_wrapping_add(count);
                let ret = unsafe {
                    self.ptr
                        .offset(D::stride_offset(&self.index, &self.strides))
                };
                let index = self.index.clone();
                self.index.last_wrapping_add(1);
                Some(IdxA!(self.inner, index, ret))
            }
        }
        #[inline(always)]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = self.len();
            (len, Some(len))
        }

        #[inline(always)]
        fn fold<Acc, G>(mut self, init: Acc, mut g: G) -> Acc
        where
            G: FnMut(Acc, Self::Item) -> Acc,
        {
            let mut accum = init;
            if self.standard_layout {
                accum = unsafe {
                    core::slice::from_raw_parts_mut(
                        self.ptr.add(self.index.last_elem()),
                        self.len(),
                    )
                    .iter_mut()
                    .fold(accum, |acc, ptr| {
                        ifIdx!(
                            g(acc, IdxA!(self.inner, self.index.clone(), ptr)),
                            self.dim.jump_index_unchecked(&mut self.index)
                        )
                    })
                };
            } else {
                let stride = self.strides.last_elem() as isize;
                let mut offset = self.index.last_elem() as isize * stride;
                while self.index.last_elem() != self.end.last_elem() {
                    unsafe {
                        accum = g(
                            accum,
                            IdxA!(self.inner, self.index.clone(), self.ptr.offset(offset)),
                        );
                    }
                    self.index.last_wrapping_add(1);
                    offset += stride;
                }
            }
            return accum;
        }
        #[inline]
        fn last(mut self) -> Option<Self::Item> {
            let ret = self.next_back();
            self.consume();
            ret
        }
    }

    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> DoubleEndedIterator
        for BaseIter1d<A, D, IDX, IdxA>
    {
        #[inline(always)]
        fn next_back(&mut self) -> Option<Self::Item> {
            if self.index.last_elem() == self.end.last_elem() {
                None
            } else {
                self.end.last_wrapping_sub(1);
                let ret = unsafe { self.ptr.offset(D::stride_offset(&self.end, &self.strides)) };
                Some(IdxA!(self.inner, self.end.clone(), ret))
            }
        }

        #[inline(always)]
        fn nth_back(&mut self, count: usize) -> Option<Self::Item> {
            if self.len() <= count {
                self.end.set_last_elem(self.index.last_elem());
                None
            } else {
                self.end.last_wrapping_sub(count + 1);
                let ret = unsafe { self.ptr.offset(D::stride_offset(&self.end, &self.strides)) };
                Some(IdxA!(self.inner, self.end.clone(), ret))
            }
        }
        #[inline(always)]
        fn rfold<Acc, G>(mut self, init: Acc, mut g: G) -> Acc
        where
            G: FnMut(Acc, Self::Item) -> Acc,
        {
            let mut accum = init;
            let stride = self.strides.last_elem() as isize;
            if self.standard_layout {
                let mut end = self.end.clone();
                self.dim.jump_index_back_unchecked(&mut end);
                accum = unsafe {
                    core::slice::from_raw_parts_mut(
                        self.ptr.add(self.index.last_elem()),
                        self.len(),
                    )
                }
                .iter_mut()
                .rfold(accum, |acc, ptr| {
                    ifIdx!(
                        g(acc, IdxA!(self.inner, end.clone(), ptr)),
                        self.dim.jump_index_back_unchecked(&mut self.end)
                    )
                });
            } else {
                let mut offset = self.end.last_elem() as isize * stride;
                offset -= stride;
                while self.index.last_elem() != self.end.last_elem() {
                    self.end.last_wrapping_sub(1);
                    unsafe {
                        accum = g(
                            accum,
                            IdxA!(self.inner, self.end.clone(), self.ptr.offset(offset)),
                        );
                    }
                    offset -= stride;
                }
            }
            return accum;
        }
    }
    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> ExactSizeIterator
        for BaseIter1d<A, D, IDX, IdxA>
    {
        #[inline(always)]
        fn len(&self) -> usize {
            self.end.last_elem() - self.index.last_elem()
        }
    }

    clone_bounds!(
        [A, D: Clone+Dimension, const IDX:bool, IdxA:BIItemT<A,D, IDX>]
        BaseIter1d[A, D,IDX, IdxA] {
            @copy {
                ptr,
            }
            dim,
            strides,
            end,
            index,
            standard_layout,
            inner,
            _item,
        }
    );
}
//====================================================================================

mod base_iter_nd {
    use super::*;

    impl<A: Debug, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> Debug
        for BaseIterNd<A, D, IDX, IdxA>
    {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("BaseIterNd")
                .field("ptr", &self.ptr)
                .field("dim", &self.dim)
                .field("index", &self.index)
                .field("end", &self.end)
                .field("strides", &self.strides)
                .field("elems_left", &self.elems_left)
                .field("elems_left_row", &self.elems_left_row)
                .field("elems_left_row_back_idx", &self.elems_left_row_back_idx)
                .field("offset_front", &self.offset_front)
                .field("offset_back", &self.offset_back)
                .field("inner", &self.inner)
                .finish()
        }
    }

    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> BaseIterNd<A, D, IDX, IdxA> {
        #[inline(always)]
        fn elems_left_row_calc(mut self) -> Self {
            if self.elems_left > self.dim.last_elem() {
                self.elems_left_row = [
                    self.dim.last_elem() - self.index.last_elem(),
                    self.end.last_elem() + 1,
                ];
                self.elems_left = self.elems_left - self.elems_left_row[0] - self.elems_left_row[1];
                self.elems_left_row_back_idx = 1;
            } else {
                self.elems_left_row = [self.elems_left, 0];
                self.elems_left = 0;
                self.elems_left_row_back_idx = 0;
            }
            self
        }
        /// Creating a Baseiter is unsafe because shape and stride parameters need
        /// to be correct to avoid performing an unsafe pointer offset while
        /// iterating.
        #[inline(always)]
        pub unsafe fn new(ptr: *mut A, len: D, strides: D, inner: IdxA::Inner) -> Self {
            let (_, mut elem_count, len, strides) = if IdxA::REQUIRES_IDX {
                (false, len.size(), len, strides)
            } else {
                len.dim_stride_analysis(strides)
            };
            let (elems_left_row, elems_left_row_back_idx);
            let mut end = len.clone();
            (0..len.ndim()).for_each(|i| unsafe {
                *end.slice_mut().get_unchecked_mut(i) =
                    len.slice().get_unchecked(i).wrapping_sub(1);
            });
            if elem_count > len.last_elem() && len.ndim() != 0 {
                //elem_count is multiple of len.last_elem()
                elem_count -= 2 * len.last_elem();
                elems_left_row = [len.last_elem(); 2];
                elems_left_row_back_idx = 1;
            } else {
                elems_left_row = [elem_count, 0];
                elems_left_row_back_idx = 0;
                elem_count = 0;
            }
            let offset_back = D::stride_offset(&end, &strides);
            BaseIterNd {
                ptr,
                index: D::zeros(len.ndim()),
                dim: len,
                strides,
                end,
                elems_left: elem_count,
                elems_left_row,
                elems_left_row_back_idx,
                offset_front: 0,
                offset_back,
                inner,
                _item: PhantomData,
            }
        }
        /// Splits the iterator at `index`, yielding two disjoint iterators.
        ///
        /// `index` is relative to the current state of the iterator (which is not
        /// necessarily the start of the axis).
        ///
        /// **Panics** if `index` is strictly greater than the iterator's remaining
        /// length.
        #[inline(always)]
        pub(crate) fn split_at(self, index: usize) -> (Self, Self) {
            assert!(index <= self.len());
            let mut mid = self.index.clone();
            self.dim.jump_index_by_unchecked(&mut mid, index);
            let mut end1 = mid.clone();
            self.dim.jump_index_back_unchecked(&mut end1);
            let (offset_back, offset_front) = (
                D::stride_offset(&end1, &self.strides),
                D::stride_offset(&mid, &self.strides),
            );
            let right_elems_left = self.len() - index;
            let left = BaseIterNd {
                index: self.index,
                dim: self.dim.clone(),
                strides: self.strides.clone(),
                ptr: self.ptr,
                end: end1,
                elems_left: index,
                elems_left_row: [0; 2],
                elems_left_row_back_idx: 0,
                offset_front: self.offset_front,
                offset_back,
                inner: self.inner.clone(),
                _item: self._item,
            }
            .elems_left_row_calc();
            let right = BaseIterNd {
                index: mid,
                dim: self.dim,
                strides: self.strides,
                ptr: self.ptr,
                end: self.end,
                elems_left: right_elems_left,
                elems_left_row: [0; 2],
                elems_left_row_back_idx: 0,
                offset_front,
                offset_back: self.offset_back,
                inner: self.inner,
                _item: self._item,
            }
            .elems_left_row_calc();
            (left, right)
        }
        #[inline]
        pub(crate) fn consume(&mut self) {
            self.elems_left = 0;
            self.elems_left_row = [0, 0];
            self.elems_left_row_back_idx = 0;
        }
    }

    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> Iterator
        for BaseIterNd<A, D, IDX, IdxA>
    {
        type Item = IdxA::BIItem;

        #[inline(always)]
        fn next(&mut self) -> Option<Self::Item> {
            let stride = (self.strides.last_elem() as Ixs) as isize;
            BaseIterNdNext!(
                self,
                stride,
                index,
                last_wrapping_add,
                jump_h_index_unchecked,
                0,
                0,
                true,
                offset_front,
                stride_h_offset
            )
        }
        #[inline(always)]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = self.len();
            (len, Some(len))
        }

        #[inline(always)]
        fn fold<Acc, G>(mut self, init: Acc, mut g: G) -> Acc
        where
            G: FnMut(Acc, Self::Item) -> Acc,
        {
            let stride = self.strides.last_elem() as isize;
            BaseIterNdFold!(
                nSEF,
                self,
                init,
                g,
                stride,
                index,
                last_wrapping_add,
                jump_h_index_unchecked,
                0,
                {},
                fold,
                offset_front,
                stride_h_offset
            );
        }
        #[inline]
        fn last(mut self) -> Option<Self::Item> {
            let ret = self.next_back();
            self.consume();
            ret
        }
    }

    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> DoubleEndedIterator
        for BaseIterNd<A, D, IDX, IdxA>
    {
        #[inline(always)]
        fn next_back(&mut self) -> Option<Self::Item> {
            let stride = -(self.strides.last_elem() as Ixs) as isize;
            BaseIterNdNext!(
                self,
                stride,
                end,
                last_wrapping_sub,
                jump_h_index_back_unchecked,
                self.dim.last_elem() - 1,
                self.elems_left_row_back_idx,
                false,
                offset_back,
                stride_offset
            )
        }

        #[inline(always)]
        fn rfold<Acc, G>(mut self, init: Acc, mut g: G) -> Acc
        where
            G: FnMut(Acc, Self::Item) -> Acc,
        {
            let stride = -(self.strides.last_elem() as isize);
            BaseIterNdFold!(
                nSEF,
                self,
                init,
                g,
                stride,
                end,
                last_wrapping_sub,
                jump_h_index_back_unchecked,
                self.dim.last_elem() - 1,
                {
                    self.elems_left_row[0] = self.elems_left_row[self.elems_left_row_back_idx];
                    self.end.set_last_elem(self.dim.last_elem() - 1);
                },
                rfold,
                offset_back,
                stride_offset
            );
        }
    }
    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> ExactSizeIterator
        for BaseIterNd<A, D, IDX, IdxA>
    {
        #[inline(always)]
        fn len(&self) -> usize {
            self.elems_left + self.elems_left_row[0] + self.elems_left_row[1]
        }
    }
    clone_bounds!(
        [A, D: Clone+Dimension, const IDX:bool, IdxA: BIItemT<A, D,IDX>]
        BaseIterNd[A, D, IDX, IdxA] {
            @copy {
                ptr,
            }
            dim,
            strides,
            end,
            elems_left,
            index,
            elems_left_row,
            elems_left_row_back_idx,
            offset_front,
            offset_back,
            inner,
            _item,
        }
    );
}
//====================================================================================

mod base_iter_nd_sef {
    use super::*;
    impl<A: Debug, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> Debug
        for BaseIterNdSEF<A, D, IDX, IdxA>
    {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("BaseIterNdSEF")
                .field("ptr", &self.ptr)
                .field("dim", &self.dim)
                .field("index", &self.index)
                .field("strides", &self.strides)
                .field("elems_left", &self.elems_left)
                .field("elems_left_row", &self.elems_left_row)
                .field("offset_front", &self.offset_front)
                .field("inner", &self.inner)
                .finish()
        }
    }
    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> BaseIterNdSEF<A, D, IDX, IdxA> {
        /// Creating a Baseiter is unsafe because shape and stride parameters need
        /// to be correct to avoid performing an unsafe pointer offset while
        /// iterating.
        #[inline(always)]
        pub unsafe fn new(ptr: *mut A, len: D, strides: D, inner: IdxA::Inner) -> Self {
            let mut elem_count = len.size();
            let standard_layout = if IdxA::REQUIRES_IDX {
                false
            } else {
                len.is_layout_c_unchecked(&strides)
            };
            let elems_left_row = if standard_layout {
                [elem_count]
            } else {
                [len.last_elem()]
            };
            elem_count -= elems_left_row[0];
            BaseIterNdSEF {
                ptr,
                index: D::zeros(len.ndim()),
                dim: len,
                strides,
                elems_left: elem_count,
                elems_left_row,
                offset_front: 0,
                inner,
                _item: PhantomData,
            }
        }

        #[inline]
        pub(crate) fn consume(&mut self) {
            self.elems_left = 0;
            self.elems_left_row[0] = 0;
        }
    }

    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> Iterator
        for BaseIterNdSEF<A, D, IDX, IdxA>
    {
        type Item = IdxA::BIItem;

        #[inline(always)]
        fn next(&mut self) -> Option<Self::Item> {
            let stride = (self.strides.last_elem() as Ixs) as isize;
            BaseIterNdNext!(
                SEF,
                self,
                stride,
                index,
                last_wrapping_add,
                jump_h_index_unchecked,
                0,
                0,
                true,
                offset_front,
                stride_h_offset
            )
        }
        #[inline(always)]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = self.len();
            (len, Some(len))
        }

        #[inline(always)]
        fn fold<Acc, G>(mut self, init: Acc, mut g: G) -> Acc
        where
            G: FnMut(Acc, Self::Item) -> Acc,
        {
            let stride = self.strides.last_elem() as isize;
            BaseIterNdFold!(
                self,
                init,
                g,
                stride,
                index,
                last_wrapping_add,
                jump_h_index_unchecked,
                0,
                {},
                fold,
                offset_front,
                stride_h_offset
            );
        }
        #[inline]
        fn last(mut self) -> Option<Self::Item> {
            if self.len() != 0 {
                let mut last_index = self.dim.clone();
                last_index
                    .slice_mut()
                    .iter_mut()
                    .for_each(|x| *x = x.saturating_sub(1));
                self.consume();
                let offset = D::stride_offset(&last_index, &self.strides);
                Some(unsafe { IdxA!(self.inner, last_index, self.ptr.offset(offset)) })
            } else {
                None
            }
        }
    }

    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> ExactSizeIterator
        for BaseIterNdSEF<A, D, IDX, IdxA>
    {
        #[inline(always)]
        fn len(&self) -> usize {
            self.elems_left + self.elems_left_row[0]
        }
    }
    clone_bounds!(
        [A, D: Clone+Dimension, const IDX:bool, IdxA: BIItemT<A, D,IDX>]
        BaseIterNdSEF[A, D, IDX, IdxA] {
            @copy {
                ptr,
            }
            dim,
            strides,
            elems_left,
            elems_left_row,
            index,
            offset_front,
            inner,
            _item,
        }
    );
}
//=================================================================================================

mod base_iter {
    use super::*;
    impl<A, D: Dimension, const IDX: bool, const SEF: bool, IdxA: BIItemT<A, D, IDX>> Clone
        for BaseIter<A, D, IDX, SEF, IdxA>
    {
        fn clone(&self) -> Self {
            eitherBIwrapped!(self,inner => inner.clone())
        }
    }

    impl<A: Debug, D: Dimension, const IDX: bool, const SEF: bool, IdxA: BIItemT<A, D, IDX>> Debug
        for BaseIter<A, D, IDX, SEF, IdxA>
    {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str("{")?;
            f.write_str("Iter type: ")?;
            f.write_str(IdxA::NAME)?;
            f.write_str("; Iter core: ")?;
            eitherBI!(self,inner=> inner.fmt(f))?;
            f.write_str("}")
        }
    }
    impl<A, D: Dimension, const IDX: bool, const SEF: bool, IdxA: BIItemT<A, D, IDX>>
        BaseIter<A, D, IDX, SEF, IdxA>
    {
        /// Creating a Baseiter is unsafe because shape and stride parameters need
        /// to be correct to avoid performing an unsafe pointer offset while
        /// iterating.
        #[inline(always)]
        pub unsafe fn new(ptr: *mut A, len: D, strides: D, inner: IdxA::Inner) -> Self {
            match D::NDIM {
                Some(0) => Self::D0(BaseIter0d::new(ptr, inner)),
                Some(1) => Self::D1(BaseIter1d::new(ptr, len, strides, inner)),
                _ => {
                    if SEF {
                        Self::DnSEF(BaseIterNdSEF::new(ptr, len, strides, inner))
                    } else {
                        Self::Dn(BaseIterNd::new(ptr, len, strides, inner))
                    }
                }
            }
        }
    }
    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> BaseIter<A, D, IDX, false, IdxA> {
        /// Splits the iterator at `index`, yielding two disjoint iterators.
        ///
        /// `index` is relative to the current state of the iterator (which is not
        /// necessarily the start of the axis).
        ///
        /// **Panics** if `index` is strictly greater than the iterator's remaining
        /// length.
        #[inline]
        pub fn split_at(self, index: usize) -> (Self, Self) {
            match D::NDIM {
                Some(0) => {
                    let ret = unwrapBI!(self,D0,inner => inner.split_at(index));
                    (Self::D0(ret.0), Self::D0(ret.1))
                }
                Some(1) => {
                    let ret = unwrapBI!(self,D1,inner => inner.split_at(index));
                    (Self::D1(ret.0), Self::D1(ret.1))
                }
                _ => {
                    let ret = unwrapBI!(self,Dn,inner => inner.split_at(index));
                    (Self::Dn(ret.0), Self::Dn(ret.1))
                }
            }
        }
    }

    impl<A, D: Dimension, const IDX: bool, const SEF: bool, IdxA: BIItemT<A, D, IDX>> Iterator
        for BaseIter<A, D, IDX, SEF, IdxA>
    {
        type Item = IdxA::BIItem;

        #[inline(always)]
        fn next(&mut self) -> Option<Self::Item> {
            eitherBI!(self,inner=>inner.next())
        }
        #[inline(always)]
        fn nth(&mut self, count: usize) -> Option<Self::Item> {
            eitherBI!(self,inner=>inner.nth(count))
        }
        #[inline(always)]
        fn size_hint(&self) -> (usize, Option<usize>) {
            eitherBI!(self,inner=>inner.size_hint())
        }

        #[inline]
        fn fold<Acc, G>(self, init: Acc, g: G) -> Acc
        where
            G: FnMut(Acc, Self::Item) -> Acc,
        {
            eitherBI!(self,inner=>inner.fold(init,g))
        }

        #[inline(always)]
        fn count(mut self) -> usize
        where
            Self: Sized,
        {
            let ret = self.len();
            eitherBI!(&mut self,inner=>inner.consume());
            ret
        }

        #[inline]
        fn last(self) -> Option<Self::Item> {
            eitherBI!(self,inner=>inner.last())
        }
    }

    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> DoubleEndedIterator
        for BaseIter<A, D, IDX, false, IdxA>
    {
        #[inline(always)]
        fn next_back(&mut self) -> Option<Self::Item> {
            eitherBI!(nSEF, self,inner=>inner.next_back())
        }
        #[inline(always)]
        fn nth_back(&mut self, count: usize) -> Option<Self::Item> {
            eitherBI!(nSEF, self,inner=>inner.nth_back(count))
        }
        #[inline]
        fn rfold<Acc, G>(self, init: Acc, g: G) -> Acc
        where
            G: FnMut(Acc, Self::Item) -> Acc,
        {
            eitherBI!(nSEF, self,inner=>inner.rfold(init,g))
        }
    }
    impl<A, D: Dimension, const IDX: bool, const SEF: bool, IdxA: BIItemT<A, D, IDX>>
        ExactSizeIterator for BaseIter<A, D, IDX, SEF, IdxA>
    {
        #[inline(always)]
        fn len(&self) -> usize {
            eitherBI!(self,inner=>inner.len())
        }
    }
    // impl<A, D: Dimension, IdxA: BIItemT<A, D, false>> BaseIter<A, D, false, IdxA> {
    //     #[inline]
    //     pub fn to_indexed<IdxB: BIItemT<A, D, true>>(mut self) -> BaseIter<A, D, true, IdxB> {
    //         unsafe { (self.borrow_mut() as *mut _ as *mut BaseIter<A, D, true, IdxB>).read() }
    //     }
    // }
    // impl<A, D: Dimension, IdxA: BIItemT<A, D, true>> BaseIter<A, D, true, IdxA> {
    //     #[inline]
    //     pub fn to_unindexed<IdxB: BIItemT<A, D, false>>(mut self) -> BaseIter<A, D, false, IdxB> {
    //         unsafe { (self.borrow_mut() as *mut _ as *mut BaseIter<A, D, false, IdxB>).read() }
    //     }
    // }
}
impl<A, const IDX: bool, IdxA: BIItemT<A, Ix1, IDX>> NdProducer
    for BaseIter<A, Ix1, IDX, false, IdxA>
{
    type Item = <Self as Iterator>::Item;
    type Dim = Ix1;
    type Ptr = *mut A;
    type Stride = isize;

    fn layout(&self) -> crate::Layout {
        crate::Layout::one_dimensional()
    }

    fn raw_dim(&self) -> Self::Dim {
        Ix1(self.len())
    }

    fn as_ptr(&self) -> Self::Ptr {
        if self.len() > 0 {
            // `self.iter.index` is guaranteed to be in-bounds if any of the
            // iterator remains (i.e. if `self.len() > 0`).
            unwrapBI!(self,D1,inner=>unsafe { inner.ptr.offset(Ix1::stride_offset(&inner.index, &inner.strides)) })
        } else {
            // In this case, `self.iter.index` may be past the end, so we must
            // not call `.offset()`. It's okay to return a dangling pointer
            // because it will never be used in the length 0 case.
            std::ptr::NonNull::dangling().as_ptr()
        }
    }

    fn contiguous_stride(&self) -> Self::Stride {
        unwrapBI!(self, D1).strides.last_elem() as isize
    }

    unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
        IdxA!(
            unwrapBI!(self, D1).inner,
            unwrapBI!(self, D1).index.clone(),
            ptr
        )
    }

    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
        unwrapBI!(self,D1,inner=>unsafe { inner.ptr.offset(Ix1::stride_offset(&Ix1(inner.index[0]+i[0]), &inner.strides)) })
    }

    fn stride_of(&self, _axis: Axis) -> isize {
        self.contiguous_stride()
    }

    fn split_at(self, _axis: Axis, index: usize) -> (Self, Self) {
        self.split_at(index)
    }

    private_impl! {}
}

//=================================================================================================
pub use base_iter::*;

pub type PtrIter<A, D> = BaseIter<A, D, false, false, BIItemPtr<A>>;
/// An iterator over the elements of an array.
///
/// Iterator element type is `&'a A`.
///
/// See [`.iter()`](ArrayBase::iter) for more information.
pub type Iter<'a, A, D> = BaseIter<A, D, false, false, BIItemRef<'a, A>>;

/// An iterator over the elements of an array (mutable).
///
/// Iterator element type is `&'a mut A`.
///
/// See [`.iter_mut()`](ArrayBase::iter_mut) for more information.
pub type IterMut<'a, A, D> = BaseIter<A, D, false, false, BIItemRefMut<'a, A>>;

/// An iterator over the indexes and elements of an array.
///
/// See [`.indexed_iter()`](ArrayBase::indexed_iter) for more information.
pub type IndexedIter<'a, A, D> = BaseIter<A, D, true, false, BIItemRef<'a, A>>;

/// An iterator over the indexes and elements of an array (mutable).
///
/// See [`.indexed_iter_mut()`](ArrayBase::indexed_iter_mut) for more information.
pub type IndexedIterMut<'a, A, D> = BaseIter<A, D, true, false, BIItemRefMut<'a, A>>;

/// An iterator that traverses over all axes but one, and yields a view for
/// each lane along that axis.
///
/// See [`.lanes()`](ArrayBase::lanes) for more information.
pub type LanesIter<'a, A, D> = BaseIter<A, D, false, false, BIItemArrayView<'a, A, Ix1>>;

/// An iterator that traverses over all dimensions but the innermost,
/// and yields each inner row (mutable).
///
/// See [`.lanes_mut()`](ArrayBase::lanes_mut)
/// for more information.
pub type LanesIterMut<'a, A, D> = BaseIter<A, D, false, false, BIItemArrayViewMut<'a, A, Ix1>>;

/// Exact chunks iterator.
///
/// See [`.exact_chunks()`](ArrayBase::exact_chunks) for more
/// information.
pub type ExactChunksIter<'a, A, D> = BaseIter<A, D, false, false, BIItemArrayView<'a, A, D>>;

/// Exact chunks iterator.
///
/// See [`.exact_chunks_mut()`](ArrayBase::exact_chunks_mut)
/// for more information.
pub type ExactChunksIterMut<'a, A, D> = BaseIter<A, D, false, false, BIItemArrayViewMut<'a, A, D>>;

/// Window iterator.
///
/// See [`.windows()`](ArrayBase::windows) for more
/// information.
pub type WindowsIter<'a, A, D> = BaseIter<A, D, false, false, BIItemArrayView<'a, A, D>>;

/// An iterator that traverses over an axis and
/// and yields each subview.
///
/// The outermost dimension is `Axis(0)`, created with `.outer_iter()`,
/// but you can traverse arbitrary dimension with `.axis_iter()`.
///
/// For example, in a 3 × 5 × 5 array, with `axis` equal to `Axis(2)`,
/// the iterator element is a 3 × 5 subview (and there are 5 in total).
///
/// Iterator element type is `ArrayView<'a, A, D>`.
///
/// See [`.outer_iter()`](ArrayBase::outer_iter)
/// or [`.axis_iter()`](ArrayBase::axis_iter)
/// for more information.
pub type AxisIter<'a, A, DI> = BaseIter<A, Ix1, false, false, BIItemArrayView<'a, A, DI>>;

/// An iterator that traverses over an axis and
/// and yields each subview (mutable)
///
/// The outermost dimension is `Axis(0)`, created with `.outer_iter()`,
/// but you can traverse arbitrary dimension with `.axis_iter()`.
///
/// For example, in a 3 × 5 × 5 array, with `axis` equal to `Axis(2)`,
/// the iterator element is a 3 × 5 subview (and there are 5 in total).
///
/// Iterator element type is `ArrayViewMut<'a, A, D>`.
///
/// See [`.outer_iter_mut()`](ArrayBase::outer_iter_mut)
/// or [`.axis_iter_mut()`](ArrayBase::axis_iter_mut)
/// for more information.
pub type AxisIterMut<'a, A, DI> = BaseIter<A, Ix1, false, false, BIItemArrayViewMut<'a, A, DI>>;

/// An iterator that traverses over the specified axis
/// and yields views of the specified size on this axis.
///
/// For example, in a 2 × 8 × 3 array, if the axis of iteration
/// is 1 and the chunk size is 2, the yielded elements
/// are 2 × 2 × 3 views (and there are 4 in total).
///
/// Iterator element type is `ArrayView<'a, A, D>`.
///
/// See [`.axis_chunks_iter()`](ArrayBase::axis_chunks_iter) for more information.
pub type AxisChunksIter<'a, A, DI> =
    BaseIter<A, Ix1, false, false, BIItemVariableArrayView<'a, A, DI>>;

/// An iterator that traverses over the specified axis
/// and yields mutable views of the specified size on this axis.
///
/// For example, in a 2 × 8 × 3 array, if the axis of iteration
/// is 1 and the chunk size is 2, the yielded elements
/// are 2 × 2 × 3 views (and there are 4 in total).
///
/// Iterator element type is `ArrayViewMut<'a, A, D>`.
///
/// See [`.axis_chunks_iter_mut()`](ArrayBase::axis_chunks_iter_mut)
/// for more information.
pub type AxisChunksIterMut<'a, A, DI> =
    BaseIter<A, Ix1, false, false, BIItemVariableArrayViewMut<'a, A, DI>>;

//=================================================================================================

send_sync_bi!(BIItemRef, Sync);
send_sync_bi_array_view!(BIItemArrayView, Sync);
send_sync_bi_array_view!(BIItemVariableArrayView, Sync);

send_sync_bi!(BIItemRefMut, Send);
send_sync_bi_array_view!(BIItemArrayViewMut, Send);
send_sync_bi_array_view!(BIItemVariableArrayViewMut, Send);

/// (Trait used internally) An iterator that we trust
/// to deliver exactly as many items as it said it would.
///
/// The iterator must produce exactly the number of elements it reported or
/// diverge before reaching the end.
#[allow(clippy::missing_safety_doc)] // not nameable downstream
pub unsafe trait TrustedIterator {}

use crate::indexes::IndicesIterF;
use crate::iter::IndicesIter;
#[cfg(feature = "std")]
use crate::{geomspace::Geomspace, linspace::Linspace, logspace::Logspace};
#[cfg(feature = "std")]
unsafe impl<F> TrustedIterator for Linspace<F> {}
#[cfg(feature = "std")]
unsafe impl<F> TrustedIterator for Geomspace<F> {}
#[cfg(feature = "std")]
unsafe impl<F> TrustedIterator for Logspace<F> {}
unsafe impl<A, D: Dimension, const IDX: bool, const SEF: bool, IdxA: BIItemT<A, D, IDX>>
    TrustedIterator for BaseIter<A, D, IDX, SEF, IdxA>
{
}
unsafe impl<I> TrustedIterator for std::iter::Cloned<I> where I: TrustedIterator {}
unsafe impl<I, F> TrustedIterator for std::iter::Map<I, F> where I: TrustedIterator {}
unsafe impl<'a, A> TrustedIterator for slice::Iter<'a, A> {}
unsafe impl<'a, A> TrustedIterator for slice::IterMut<'a, A> {}
unsafe impl TrustedIterator for ::std::ops::Range<usize> {}
// FIXME: These indices iter are dubious -- size needs to be checked up front.
unsafe impl<D> TrustedIterator for IndicesIter<D> where D: Dimension {}
unsafe impl<D> TrustedIterator for IndicesIterF<D> where D: Dimension {}
unsafe impl<A, D> TrustedIterator for IntoIter<A, D> where D: Dimension {}

/// Like Iterator::collect, but only for trusted length iterators
pub fn to_vec<I>(iter: I) -> Vec<I::Item>
where
    I: TrustedIterator + ExactSizeIterator,
{
    to_vec_mapped(iter, |x| x)
}

/// Like Iterator::collect, but only for trusted length iterators
pub fn to_vec_mapped<I, F, B>(iter: I, mut f: F) -> Vec<B>
where
    I: TrustedIterator + ExactSizeIterator,
    F: FnMut(I::Item) -> B,
{
    // Use an `unsafe` block to do this efficiently.
    // We know that iter will produce exactly .size() elements,
    // and the loop can vectorize if it's clean (without branch to grow the vector).
    let (size, _) = iter.size_hint();
    let mut result = Vec::with_capacity(size);
    let mut out_ptr = result.as_mut_ptr();
    iter.fold((), |(), elt| unsafe {
        ptr::write(out_ptr, f(elt));
        out_ptr = out_ptr.offset(1);
    });
    unsafe { result.set_len(size) };
    debug_assert_eq!(size, result.len());
    result
}
