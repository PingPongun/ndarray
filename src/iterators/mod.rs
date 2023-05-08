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

use super::{ArrayBase, ArrayView, ArrayViewMut, Axis, Data, NdProducer, RemoveAxis};
use super::{Dimension, Ix, Ixs};

pub use self::chunks::{ExactChunks, ExactChunksIter, ExactChunksIterMut, ExactChunksMut};
pub use self::into_iter::IntoIter;
pub use self::lanes::{Lanes, LanesMut};
pub use self::windows::Windows;

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
    standard_layout: bool,
    elems_left_row: [usize; 2],
    elems_left_row_back_idx: usize,
    offset_front: isize,
    offset_back: isize,
    inner: IdxA::Inner,
    _item: PhantomData<IdxA>,
}
/// Base for iterators over all axes.
///
/// Iterator element type is `*mut A`.
/// index and end values are only valid indices when elements_left >= 1

pub enum BaseIter<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> {
    D0(BaseIter0d<A, D, IDX, IdxA>),
    D1(BaseIter1d<A, D, IDX, IdxA>),
    Dn(BaseIterNd<A, D, IDX, IdxA>),
}
#[macro_use]
mod _macros {
    macro_rules! eitherBI {
        ($bi:expr, $inner:pat => $result:expr) => {
            match D::NDIM {
                Some(0) => {
                    if let BaseIter::D0($inner) = $bi {
                        $result
                    } else {
                        unsafe { unreachable_unchecked() }
                    }
                }
                Some(1) => {
                    if let BaseIter::D1($inner) = $bi {
                        $result
                    } else {
                        unsafe { unreachable_unchecked() }
                    }
                }
                _ => {
                    if let BaseIter::Dn($inner) = $bi {
                        $result
                    } else {
                        unsafe { unreachable_unchecked() }
                    }
                }
            }
        };
    }

    macro_rules! eitherBIwrapped {
        ($bi:expr, $inner:pat => $result:expr) => {
            match D::NDIM {
                Some(0) => {
                    if let BaseIter::D0($inner) = $bi {
                        BaseIter::D0($result)
                    } else {
                        unsafe { unreachable_unchecked() }
                    }
                }
                Some(1) => {
                    if let BaseIter::D1($inner) = $bi {
                        BaseIter::D1($result)
                    } else {
                        unsafe { unreachable_unchecked() }
                    }
                }
                _ => {
                    if let BaseIter::Dn($inner) = $bi {
                        BaseIter::Dn($result)
                    } else {
                        unsafe { unreachable_unchecked() }
                    }
                }
            }
        };
    }
    macro_rules! ifIdx {
        ( $func:expr , $jump:expr) => {
            if IDX {
                let ret = $func;
                $jump;
                ret
            } else {
                $func
            }
        };
    }

    macro_rules! impl_BIItem {
        ( $typ:ident,$ret:ty,$inner:ident ,$pat:pat => $func:expr) => {
            impl<'a, A, D: Dimension, $typ: Dimension> BIItemT<A, D, true> for $ret {
                type BIItem = (D::Pattern, $ret);
                type Inner = ($typ, $typ);
                const W_INNER: bool = true;

                #[inline(always)]
                fn item_idx_w_inner(
                    $inner: &Self::Inner,
                    idx: D::Pattern,
                    $pat: *mut A,
                ) -> Self::BIItem {
                    (idx, $func)
                }
            }
            impl<'a, A, D: Dimension, $typ: Dimension> BIItemT<A, D, false> for $ret {
                type BIItem = $ret;
                type Inner = ($typ, $typ);
                const W_INNER: bool = true;

                #[inline(always)]
                fn item_w_inner($inner: &Self::Inner, $pat: *mut A) -> Self::BIItem {
                    $func
                }
            }
        };
        ( $ret:ty ,$pat:pat => $func:expr) => {
            impl<'a, A, D: Dimension> BIItemT<A, D, true> for $ret {
                type BIItem = (D::Pattern, $ret);
                type Inner = ();
                const W_INNER: bool = false;

                #[inline(always)]
                fn item_idx(idx: D::Pattern, $pat: *mut A) -> Self::BIItem {
                    (idx, $func)
                }
            }
            impl<'a, A, D: Dimension> BIItemT<A, D, false> for $ret {
                type BIItem = $ret;
                type Inner = ();
                const W_INNER: bool = false;

                #[inline(always)]
                fn item($pat: *mut A) -> Self::BIItem {
                    $func
                }
            }
        };
    }
    macro_rules! _Idx {
        ($inner:expr,$idx:expr,$ptr:expr) => {
            if IdxA::W_INNER == false {
                IdxA::item_idx($idx, $ptr)
            } else {
                IdxA::item_idx_w_inner(&$inner, $idx, $ptr)
            }
        };
    }
    macro_rules! _nIdx {
        ($inner:expr,$ptr:expr) => {
            if IdxA::W_INNER == false {
                IdxA::item($ptr)
            } else {
                IdxA::item_w_inner(&$inner, $ptr)
            }
        };
    }
    macro_rules! IdxA {
        ($inner:expr,$idx:expr,$ptr:expr,$idx_expr:expr,$nidx_expr:expr) => {
            if IDX {
                $idx_expr;
                _Idx!($inner, $idx.into_pattern(), $ptr)
            } else {
                $nidx_expr;
                _nIdx!($inner, $ptr)
            }
        };
        ($inner:expr,$idx:expr,$ptr:expr) => {
            IdxA!($inner, $idx, $ptr, {}, {})
        };
        ($inner:expr,$idx:expr,$ptr:expr,$idx_expr:expr) => {
            IdxA!($inner, $idx, $ptr, $idx_expr, {})
        };
    }
}

pub trait BIItemT<A, D: Dimension, const IDX: bool> {
    type BIItem;
    type Inner: Clone;
    const W_INNER: bool;
    #[inline(always)]
    fn item(_val: *mut A) -> Self::BIItem {
        unreachable!()
    }
    #[inline(always)]
    fn item_idx(_idx: D::Pattern, _val: *mut A) -> Self::BIItem {
        unreachable!()
    }
    #[inline(always)]
    fn item_w_inner(_inner: &Self::Inner, _val: *mut A) -> Self::BIItem {
        unreachable!()
    }
    #[inline(always)]
    fn item_idx_w_inner(_inner: &Self::Inner, _idx: D::Pattern, _val: *mut A) -> Self::BIItem {
        unreachable!()
    }
}
impl_BIItem!(*mut A,ptr =>ptr);
impl_BIItem!(&'a A,ptr => unsafe{&*ptr});
impl_BIItem!(DI,ArrayViewMut<'a,A,DI>, inner,ptr => unsafe{ArrayViewMut::new_(ptr,inner.0.clone(), inner.1.clone() )});
impl_BIItem!(DI,ArrayView<'a,A,DI>, inner,ptr => unsafe{ArrayView::new_(ptr,inner.0.clone(), inner.1.clone() )});
impl_BIItem!(&'a mut A,ptr => unsafe{&mut *ptr});

//=================================================================================================
mod base_iter_0d {
    use super::*;
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
        pub(crate) fn split_at(
            self,
            _index: usize,
        ) -> (BaseIter<A, D, IDX, IdxA>, BaseIter<A, D, IDX, IdxA>) {
            let mut right = self.clone();
            right.elems_left = 0;
            (BaseIter::D0(self), BaseIter::D0(right))
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
                Some(IdxA!(&self.inner, D::default(), self.ptr))
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
        pub(crate) fn split_at(
            self,
            index: usize,
        ) -> (BaseIter<A, D, IDX, IdxA>, BaseIter<A, D, IDX, IdxA>) {
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
            (BaseIter::D1(left), BaseIter::D1(right))
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
                Some(IdxA!(&self.inner, index, ret, {}))
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
                Some(IdxA!(&self.inner, index, ret))
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
                            g(acc, IdxA!(&self.inner, self.index.clone(), ptr)),
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
                            IdxA!(&self.inner, self.index.clone(), self.ptr.offset(offset)),
                        );
                    }
                    self.index.last_wrapping_add(1);
                    offset += stride;
                }
            }
            return accum;
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
                Some(IdxA!(&self.inner, self.end.clone(), ret))
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
                Some(IdxA!(&self.inner, self.end.clone(), ret))
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
                        g(acc, IdxA!(&self.inner, end.clone(), ptr)),
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
                            IdxA!(&self.inner, self.end.clone(), self.ptr.offset(offset)),
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
    macro_rules! BaseIterNdFoldCore {
        ($_self:ident,$accum:ident,$g:ident,$stride:ident,$idx:ident, $idx_step:ident,$idx_jump_h:ident,$idx_def:expr, $offset:ident, $offset_func:ident) => {
            $_self.elems_left -= $_self.elems_left_row[0];
            while 0 != $_self.elems_left_row[0] {
                unsafe {
                    $accum = $g(
                        $accum,
                        IdxA!(
                            &$_self.inner,
                            $_self.$idx.clone(),
                            $_self.ptr.offset($_self.$offset)
                        ),
                    );
                }
                $_self.$offset += $stride;
                $_self.elems_left_row[0] -= 1;
                if IDX {
                    $_self.$idx.$idx_step(1);
                }
            }
            $_self.dim.$idx_jump_h(&mut $_self.$idx);
            $_self.$offset = D::$offset_func(&$_self.$idx, &$_self.strides);
            if IDX {
                $_self.$idx.set_last_elem($idx_def);
            }
        };
    }
    macro_rules! BaseIterNdFold {
        ($_self:ident,$init:ident,$g:ident,$stride:ident,$idx:ident, $idx_step:ident,$idx_jump:ident,$idx_jump_h:ident,$idx_def:expr,$pre_fold:expr,$fold:ident, $offset:ident, $offset_func:ident) => {
            let mut accum = $init;
            if $_self.standard_layout {
                //TODO CHECK if for indexed iter should standard layout still be handled separetly
                accum = unsafe {
                    core::slice::from_raw_parts_mut(
                        $_self.ptr.offset($_self.offset_front),
                        $_self.elems_left,
                    )
                }
                .iter_mut()
                .$fold(accum, |acc, ptr| {
                    ifIdx!(
                        $g(acc, IdxA!(&$_self.inner, $_self.$idx.clone(), ptr)),
                        $_self.dim.$idx_jump(&mut $_self.$idx)
                    )
                });
                $_self.elems_left = 0;
            } else {
                $pre_fold;
                if !IDX {
                    $_self.$idx.set_last_elem($idx_def);
                }
                loop {
                    BaseIterNdFoldCore!(
                        $_self,
                        accum,
                        $g,
                        $stride,
                        $idx,
                        $idx_step,
                        $idx_jump_h,
                        $idx_def,
                        $offset,
                        $offset_func
                    );
                    if $_self.elems_left > $_self.dim.last_elem() {
                        $_self.elems_left_row[0] = $_self.dim.last_elem();
                        continue;
                    } else if $_self.elems_left == 0 {
                        break;
                    } else {
                        $_self.elems_left_row[0] = $_self.elems_left;
                        continue;
                        // BaseIterNdFoldCore!(
                        //     $_self,
                        //     accum,
                        //     $g,
                        //     $stride,
                        //     $idx,
                        //     $idx_step,
                        //     $idx_jump_h,
                        //     $idx_def,
                        //     $offset,
                        //     $offset_func
                        // );
                        // break;
                    }
                }
            }
            return accum;
        };
    }
    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> BaseIterNd<A, D, IDX, IdxA> {
        #[inline(always)]
        fn elems_left_row_calc(mut self) -> Self {
            if self.elems_left > self.dim.last_elem() {
                self.elems_left_row = [
                    self.dim.last_elem() - self.index.last_elem(),
                    self.end.last_elem() + 1,
                ];
                self.elems_left_row_back_idx = 1;
            } else {
                self.elems_left_row = [self.elems_left, 0];
                self.elems_left_row_back_idx = 0;
            }
            self
        }
        /// Creating a Baseiter is unsafe because shape and stride parameters need
        /// to be correct to avoid performing an unsafe pointer offset while
        /// iterating.
        #[inline(always)]
        pub unsafe fn new(ptr: *mut A, len: D, strides: D, inner: IdxA::Inner) -> Self {
            let elem_count = len.size();
            let mut end = len.clone();
            end.slice_mut()
                .iter_mut()
                .for_each(|x| *x = x.wrapping_sub(1));
            let standard_layout = len.is_layout_c_unchecked(&strides);
            let offset_back = D::stride_offset(&end, &strides);
            BaseIterNd {
                ptr,
                index: D::zeros(len.ndim()),
                dim: len,
                strides,
                end,
                elems_left: elem_count,
                standard_layout,
                elems_left_row: [0; 2],
                elems_left_row_back_idx: 0,
                offset_front: 0,
                offset_back,
                inner,
                _item: PhantomData,
            }
            .elems_left_row_calc()
        }
        /// Splits the iterator at `index`, yielding two disjoint iterators.
        ///
        /// `index` is relative to the current state of the iterator (which is not
        /// necessarily the start of the axis).
        ///
        /// **Panics** if `index` is strictly greater than the iterator's remaining
        /// length.
        #[inline(always)]
        pub(crate) fn split_at(
            self,
            index: usize,
        ) -> (BaseIter<A, D, IDX, IdxA>, BaseIter<A, D, IDX, IdxA>) {
            assert!(index <= self.len());
            let mut mid = self.index.clone();
            self.dim.jump_index_by_unchecked(&mut mid, index);
            let mut end1 = mid.clone();
            self.dim.jump_index_back_unchecked(&mut end1);
            let (offset_back, offset_front) = (
                D::stride_offset(&end1, &self.strides),
                D::stride_offset(&mid, &self.strides),
            );
            let left = BaseIterNd {
                index: self.index,
                dim: self.dim.clone(),
                strides: self.strides.clone(),
                ptr: self.ptr,
                end: end1,
                elems_left: index,
                standard_layout: self.standard_layout,
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
                elems_left: self.elems_left - index,
                standard_layout: self.standard_layout,
                elems_left_row: [0; 2],
                elems_left_row_back_idx: 0,
                offset_front,
                offset_back: self.offset_back,
                inner: self.inner,
                _item: self._item,
            }
            .elems_left_row_calc();
            (BaseIter::Dn(left), BaseIter::Dn(right))
        }
    }

    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> Iterator
        for BaseIterNd<A, D, IDX, IdxA>
    {
        type Item = IdxA::BIItem;

        #[inline(always)]
        fn next(&mut self) -> Option<Self::Item> {
            if self.elems_left_row[0] == 0 {
                None
            } else {
                self.elems_left -= 1;
                let index = self.index.clone();
                let ret = unsafe { self.ptr.offset(self.offset_front) };
                if self.elems_left_row[0] == 1 {
                    //last element in row
                    if self.elems_left > self.dim.last_elem() {
                        self.elems_left_row[0] = self.dim.last_elem();
                    } else {
                        //starting last row
                        self.elems_left_row[0] = self.elems_left;
                        self.elems_left_row_back_idx = 0;
                    }
                    self.dim.jump_h_index_unchecked(&mut self.index); //switch to new row
                    self.index.set_last_elem(0);
                    self.offset_front = D::stride_h_offset(&self.index, &self.strides);
                    Some(IdxA!(&self.inner, index, ret))
                } else {
                    //normal(not last in row) element
                    self.elems_left_row[0] -= 1;
                    self.offset_front += (self.strides.last_elem() as Ixs) as isize;
                    self.index.last_wrapping_add(1);
                    Some(IdxA!(&self.inner, index, ret))
                }
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
            let stride = self.strides.last_elem() as isize;
            BaseIterNdFold!(
                self,
                init,
                g,
                stride,
                index,
                last_wrapping_add,
                jump_index_unchecked,
                jump_h_index_unchecked,
                0,
                {},
                fold,
                offset_front,
                stride_h_offset
            );
        }
    }

    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> DoubleEndedIterator
        for BaseIterNd<A, D, IDX, IdxA>
    {
        #[inline(always)]
        fn next_back(&mut self) -> Option<Self::Item> {
            if self.elems_left_row[self.elems_left_row_back_idx] == 0 {
                None
            } else {
                self.elems_left -= 1;
                let index = self.end.clone();
                let ret = unsafe { self.ptr.offset(self.offset_back) };
                if self.elems_left_row[self.elems_left_row_back_idx] == 1 {
                    if self.elems_left > self.dim.last_elem() {
                        self.elems_left_row[self.elems_left_row_back_idx] = self.dim.last_elem();
                    } else {
                        //last row
                        self.elems_left_row[0] = self.elems_left;
                        self.elems_left_row_back_idx = 0;
                    }
                    self.dim.jump_h_index_back_unchecked(&mut self.end); //switch to new row
                    self.end.set_last_elem(self.dim.last_elem() - 1);
                    self.offset_back = D::stride_offset(&self.end, &self.strides);
                    Some(IdxA!(&self.inner, index, ret))
                } else {
                    self.elems_left_row[self.elems_left_row_back_idx] -= 1;
                    self.offset_back -= (self.strides.last_elem() as Ixs) as isize;
                    self.end.last_wrapping_sub(1);
                    Some(IdxA!(&self.inner, index, ret))
                }
            }
        }

        #[inline(always)]
        fn rfold<Acc, G>(mut self, init: Acc, mut g: G) -> Acc
        where
            G: FnMut(Acc, Self::Item) -> Acc,
        {
            let stride = -(self.strides.last_elem() as isize);
            BaseIterNdFold!(
                self,
                init,
                g,
                stride,
                end,
                last_wrapping_sub,
                jump_index_back_unchecked,
                jump_h_index_back_unchecked,
                self.dim.last_elem() - 1,
                {
                    self.elems_left_row[0] = self.elems_left_row[self.elems_left_row_back_idx];
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
            self.elems_left
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
            standard_layout,
            elems_left_row,
            elems_left_row_back_idx,
            offset_front,
            offset_back,
            inner,
            _item,
        }
    );
}
//=================================================================================================
mod base_iter {
    use super::*;
    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> Clone
        for BaseIter<A, D, IDX, IdxA>
    {
        fn clone(&self) -> Self {
            eitherBIwrapped!(self,inner => inner.clone())
        }
    }
    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> BaseIter<A, D, IDX, IdxA> {
        /// Creating a Baseiter is unsafe because shape and stride parameters need
        /// to be correct to avoid performing an unsafe pointer offset while
        /// iterating.
        #[inline(always)]
        pub unsafe fn new(
            ptr: *mut A,
            len: D,
            strides: D,
            inner: IdxA::Inner,
        ) -> BaseIter<A, D, IDX, IdxA> {
            match D::NDIM {
                Some(0) => Self::D0(BaseIter0d::new(ptr, inner)),
                Some(1) => Self::D1(BaseIter1d::new(ptr, len, strides, inner)),
                _ => Self::Dn(BaseIterNd::new(ptr, len, strides, inner)),
            }
        }

        /// Splits the iterator at `index`, yielding two disjoint iterators.
        ///
        /// `index` is relative to the current state of the iterator (which is not
        /// necessarily the start of the axis).
        ///
        /// **Panics** if `index` is strictly greater than the iterator's remaining
        /// length.
        #[inline]
        pub(crate) fn split_at(self, index: usize) -> (Self, Self) {
            eitherBI!(self,inner=>inner.split_at(index))
        }
    }

    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> Iterator
        for BaseIter<A, D, IDX, IdxA>
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
        #[inline]
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
    }

    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> DoubleEndedIterator
        for BaseIter<A, D, IDX, IdxA>
    where
        D: Dimension,
    {
        #[inline(always)]
        fn next_back(&mut self) -> Option<Self::Item> {
            eitherBI!(self,inner=>inner.next_back())
        }
        #[inline(always)]
        fn nth_back(&mut self, count: usize) -> Option<Self::Item> {
            eitherBI!(self,inner=>inner.nth_back(count))
        }
        #[inline]
        fn rfold<Acc, G>(self, init: Acc, g: G) -> Acc
        where
            G: FnMut(Acc, Self::Item) -> Acc,
        {
            eitherBI!(self,inner=>inner.rfold(init,g))
        }
    }
    impl<A, D: Dimension, const IDX: bool, IdxA: BIItemT<A, D, IDX>> ExactSizeIterator
        for BaseIter<A, D, IDX, IdxA>
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
//=================================================================================================
pub use base_iter::*;

/// An iterator over the elements of an array.
///
/// Iterator element type is `&'a A`.
///
/// See [`.iter()`](ArrayBase::iter) for more information.
pub type Iter<'a, A, D> = BaseIter<A, D, false, &'a A>;

/// An iterator over the elements of an array (mutable).
///
/// Iterator element type is `&'a mut A`.
///
/// See [`.iter_mut()`](ArrayBase::iter_mut) for more information.
pub type IterMut<'a, A, D> = BaseIter<A, D, false, &'a mut A>;

/// An iterator over the indexes and elements of an array.
///
/// See [`.indexed_iter()`](ArrayBase::indexed_iter) for more information.
pub type IndexedIter<'a, A, D> = BaseIter<A, D, true, &'a A>;

/// An iterator over the indexes and elements of an array (mutable).
///
/// See [`.indexed_iter_mut()`](ArrayBase::indexed_iter_mut) for more information.
pub type IndexedIterMut<'a, A, D> = BaseIter<A, D, true, &'a mut A>;

/// An iterator that traverses over all axes but one, and yields a view for
/// each lane along that axis.
///
/// See [`.lanes()`](ArrayBase::lanes) for more information.
pub type LanesIter<'a, A, D> = BaseIter<A, D, false, ArrayView<'a, A, Ix1>>;

/// An iterator that traverses over all dimensions but the innermost,
/// and yields each inner row (mutable).
///
/// See [`.lanes_mut()`](ArrayBase::lanes_mut)
/// for more information.
pub type LanesIterMut<'a, A, D> = BaseIter<A, D, false, ArrayViewMut<'a, A, Ix1>>;

//=================================================================================================

#[derive(Debug)]
pub struct AxisIterCore<A, D> {
    /// Index along the axis of the value of `.next()`, relative to the start
    /// of the axis.
    index: Ix,
    /// (Exclusive) upper bound on `index`. Initially, this is equal to the
    /// length of the axis.
    end: Ix,
    /// Stride along the axis (offset between consecutive pointers).
    stride: Ixs,
    /// Shape of the iterator's items.
    inner_dim: D,
    /// Strides of the iterator's items.
    inner_strides: D,
    /// Pointer corresponding to `index == 0`.
    ptr: *mut A,
}

clone_bounds!(
    [A, D: Clone]
    AxisIterCore[A, D] {
        @copy {
            index,
            end,
            stride,
            ptr,
        }
        inner_dim,
        inner_strides,
    }
);

impl<A, D: Dimension> AxisIterCore<A, D> {
    /// Constructs a new iterator over the specified axis.
    fn new<S, Di>(v: ArrayBase<S, Di>, axis: Axis) -> Self
    where
        Di: RemoveAxis<Smaller = D>,
        S: Data<Elem = A>,
    {
        AxisIterCore {
            index: 0,
            end: v.len_of(axis),
            stride: v.stride_of(axis),
            inner_dim: v.dim.remove_axis(axis),
            inner_strides: v.strides.remove_axis(axis),
            ptr: v.ptr.as_ptr(),
        }
    }

    #[inline]
    unsafe fn offset(&self, index: usize) -> *mut A {
        debug_assert!(
            index < self.end,
            "index={}, end={}, stride={}",
            index,
            self.end,
            self.stride
        );
        self.ptr.offset(index as isize * self.stride)
    }

    /// Splits the iterator at `index`, yielding two disjoint iterators.
    ///
    /// `index` is relative to the current state of the iterator (which is not
    /// necessarily the start of the axis).
    ///
    /// **Panics** if `index` is strictly greater than the iterator's remaining
    /// length.
    fn split_at(self, index: usize) -> (Self, Self) {
        assert!(index <= self.len());
        let mid = self.index + index;
        let left = AxisIterCore {
            index: self.index,
            end: mid,
            stride: self.stride,
            inner_dim: self.inner_dim.clone(),
            inner_strides: self.inner_strides.clone(),
            ptr: self.ptr,
        };
        let right = AxisIterCore {
            index: mid,
            end: self.end,
            stride: self.stride,
            inner_dim: self.inner_dim,
            inner_strides: self.inner_strides,
            ptr: self.ptr,
        };
        (left, right)
    }

    /// Does the same thing as `.next()` but also returns the index of the item
    /// relative to the start of the axis.
    fn next_with_index(&mut self) -> Option<(usize, *mut A)> {
        let index = self.index;
        self.next().map(|ptr| (index, ptr))
    }

    /// Does the same thing as `.next_back()` but also returns the index of the
    /// item relative to the start of the axis.
    fn next_back_with_index(&mut self) -> Option<(usize, *mut A)> {
        self.next_back().map(|ptr| (self.end, ptr))
    }
}

impl<A, D> Iterator for AxisIterCore<A, D>
where
    D: Dimension,
{
    type Item = *mut A;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            None
        } else {
            let ptr = unsafe { self.offset(self.index) };
            self.index += 1;
            Some(ptr)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<A, D> DoubleEndedIterator for AxisIterCore<A, D>
where
    D: Dimension,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            None
        } else {
            let ptr = unsafe { self.offset(self.end - 1) };
            self.end -= 1;
            Some(ptr)
        }
    }
}

impl<A, D> ExactSizeIterator for AxisIterCore<A, D>
where
    D: Dimension,
{
    fn len(&self) -> usize {
        self.end - self.index
    }
}

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
#[derive(Debug)]
pub struct AxisIter<'a, A, D> {
    iter: AxisIterCore<A, D>,
    life: PhantomData<&'a A>,
}

clone_bounds!(
    ['a, A, D: Clone]
    AxisIter['a, A, D] {
        @copy {
            life,
        }
        iter,
    }
);

impl<'a, A, D: Dimension> AxisIter<'a, A, D> {
    /// Creates a new iterator over the specified axis.
    pub(crate) fn new<Di>(v: ArrayView<'a, A, Di>, axis: Axis) -> Self
    where
        Di: RemoveAxis<Smaller = D>,
    {
        AxisIter {
            iter: AxisIterCore::new(v, axis),
            life: PhantomData,
        }
    }

    /// Splits the iterator at `index`, yielding two disjoint iterators.
    ///
    /// `index` is relative to the current state of the iterator (which is not
    /// necessarily the start of the axis).
    ///
    /// **Panics** if `index` is strictly greater than the iterator's remaining
    /// length.
    pub fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.iter.split_at(index);
        (
            AxisIter {
                iter: left,
                life: self.life,
            },
            AxisIter {
                iter: right,
                life: self.life,
            },
        )
    }
}

impl<'a, A, D> Iterator for AxisIter<'a, A, D>
where
    D: Dimension,
{
    type Item = ArrayView<'a, A, D>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ptr| unsafe { self.as_ref(ptr) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, A, D> DoubleEndedIterator for AxisIter<'a, A, D>
where
    D: Dimension,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|ptr| unsafe { self.as_ref(ptr) })
    }
}

impl<'a, A, D> ExactSizeIterator for AxisIter<'a, A, D>
where
    D: Dimension,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

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
pub struct AxisIterMut<'a, A, D> {
    iter: AxisIterCore<A, D>,
    life: PhantomData<&'a mut A>,
}

impl<'a, A, D: Dimension> AxisIterMut<'a, A, D> {
    /// Creates a new iterator over the specified axis.
    pub(crate) fn new<Di>(v: ArrayViewMut<'a, A, Di>, axis: Axis) -> Self
    where
        Di: RemoveAxis<Smaller = D>,
    {
        AxisIterMut {
            iter: AxisIterCore::new(v, axis),
            life: PhantomData,
        }
    }

    /// Splits the iterator at `index`, yielding two disjoint iterators.
    ///
    /// `index` is relative to the current state of the iterator (which is not
    /// necessarily the start of the axis).
    ///
    /// **Panics** if `index` is strictly greater than the iterator's remaining
    /// length.
    pub fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.iter.split_at(index);
        (
            AxisIterMut {
                iter: left,
                life: self.life,
            },
            AxisIterMut {
                iter: right,
                life: self.life,
            },
        )
    }
}

impl<'a, A, D> Iterator for AxisIterMut<'a, A, D>
where
    D: Dimension,
{
    type Item = ArrayViewMut<'a, A, D>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ptr| unsafe { self.as_ref(ptr) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, A, D> DoubleEndedIterator for AxisIterMut<'a, A, D>
where
    D: Dimension,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|ptr| unsafe { self.as_ref(ptr) })
    }
}

impl<'a, A, D> ExactSizeIterator for AxisIterMut<'a, A, D>
where
    D: Dimension,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, A, D: Dimension> NdProducer for AxisIter<'a, A, D> {
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
            unsafe { self.iter.offset(self.iter.index) }
        } else {
            // In this case, `self.iter.index` may be past the end, so we must
            // not call `.offset()`. It's okay to return a dangling pointer
            // because it will never be used in the length 0 case.
            std::ptr::NonNull::dangling().as_ptr()
        }
    }

    fn contiguous_stride(&self) -> isize {
        self.iter.stride
    }

    unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
        ArrayView::new_(
            ptr,
            self.iter.inner_dim.clone(),
            self.iter.inner_strides.clone(),
        )
    }

    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
        self.iter.offset(self.iter.index + i[0])
    }

    fn stride_of(&self, _axis: Axis) -> isize {
        self.contiguous_stride()
    }

    fn split_at(self, _axis: Axis, index: usize) -> (Self, Self) {
        self.split_at(index)
    }

    private_impl! {}
}

impl<'a, A, D: Dimension> NdProducer for AxisIterMut<'a, A, D> {
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
            unsafe { self.iter.offset(self.iter.index) }
        } else {
            // In this case, `self.iter.index` may be past the end, so we must
            // not call `.offset()`. It's okay to return a dangling pointer
            // because it will never be used in the length 0 case.
            std::ptr::NonNull::dangling().as_ptr()
        }
    }

    fn contiguous_stride(&self) -> isize {
        self.iter.stride
    }

    unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
        ArrayViewMut::new_(
            ptr,
            self.iter.inner_dim.clone(),
            self.iter.inner_strides.clone(),
        )
    }

    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
        self.iter.offset(self.iter.index + i[0])
    }

    fn stride_of(&self, _axis: Axis) -> isize {
        self.contiguous_stride()
    }

    fn split_at(self, _axis: Axis, index: usize) -> (Self, Self) {
        self.split_at(index)
    }

    private_impl! {}
}

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
pub struct AxisChunksIter<'a, A, D> {
    iter: AxisIterCore<A, D>,
    /// Index of the partial chunk (the chunk smaller than the specified chunk
    /// size due to the axis length not being evenly divisible). If the axis
    /// length is evenly divisible by the chunk size, this index is larger than
    /// the maximum valid index.
    partial_chunk_index: usize,
    /// Dimension of the partial chunk.
    partial_chunk_dim: D,
    life: PhantomData<&'a A>,
}

clone_bounds!(
    ['a, A, D: Clone]
    AxisChunksIter['a, A, D] {
        @copy {
            life,
            partial_chunk_index,
        }
        iter,
        partial_chunk_dim,
    }
);

/// Computes the information necessary to construct an iterator over chunks
/// along an axis, given a `view` of the array, the `axis` to iterate over, and
/// the chunk `size`.
///
/// Returns an axis iterator with the correct stride to move between chunks,
/// the number of chunks, and the shape of the last chunk.
///
/// **Panics** if `size == 0`.
fn chunk_iter_parts<A, D: Dimension>(
    v: ArrayView<'_, A, D>,
    axis: Axis,
    size: usize,
) -> (AxisIterCore<A, D>, usize, D) {
    assert_ne!(size, 0, "Chunk size must be nonzero.");
    let axis_len = v.len_of(axis);
    let n_whole_chunks = axis_len / size;
    let chunk_remainder = axis_len % size;
    let iter_len = if chunk_remainder == 0 {
        n_whole_chunks
    } else {
        n_whole_chunks + 1
    };
    let stride = if n_whole_chunks == 0 {
        // This case avoids potential overflow when `size > axis_len`.
        0
    } else {
        v.stride_of(axis) * size as isize
    };

    let axis = axis.index();
    let mut inner_dim = v.dim.clone();
    inner_dim[axis] = size;

    let mut partial_chunk_dim = v.dim;
    partial_chunk_dim[axis] = chunk_remainder;
    let partial_chunk_index = n_whole_chunks;

    let iter = AxisIterCore {
        index: 0,
        end: iter_len,
        stride,
        inner_dim,
        inner_strides: v.strides,
        ptr: v.ptr.as_ptr(),
    };

    (iter, partial_chunk_index, partial_chunk_dim)
}

impl<'a, A, D: Dimension> AxisChunksIter<'a, A, D> {
    pub(crate) fn new(v: ArrayView<'a, A, D>, axis: Axis, size: usize) -> Self {
        let (iter, partial_chunk_index, partial_chunk_dim) = chunk_iter_parts(v, axis, size);
        AxisChunksIter {
            iter,
            partial_chunk_index,
            partial_chunk_dim,
            life: PhantomData,
        }
    }
}

macro_rules! chunk_iter_impl {
    ($iter:ident, $array:ident) => {
        impl<'a, A, D> $iter<'a, A, D>
        where
            D: Dimension,
        {
            fn get_subview(&self, index: usize, ptr: *mut A) -> $array<'a, A, D> {
                if index != self.partial_chunk_index {
                    unsafe {
                        $array::new_(
                            ptr,
                            self.iter.inner_dim.clone(),
                            self.iter.inner_strides.clone(),
                        )
                    }
                } else {
                    unsafe {
                        $array::new_(
                            ptr,
                            self.partial_chunk_dim.clone(),
                            self.iter.inner_strides.clone(),
                        )
                    }
                }
            }

            /// Splits the iterator at index, yielding two disjoint iterators.
            ///
            /// `index` is relative to the current state of the iterator (which is not
            /// necessarily the start of the axis).
            ///
            /// **Panics** if `index` is strictly greater than the iterator's remaining
            /// length.
            pub fn split_at(self, index: usize) -> (Self, Self) {
                let (left, right) = self.iter.split_at(index);
                (
                    Self {
                        iter: left,
                        partial_chunk_index: self.partial_chunk_index,
                        partial_chunk_dim: self.partial_chunk_dim.clone(),
                        life: self.life,
                    },
                    Self {
                        iter: right,
                        partial_chunk_index: self.partial_chunk_index,
                        partial_chunk_dim: self.partial_chunk_dim,
                        life: self.life,
                    },
                )
            }
        }

        impl<'a, A, D> Iterator for $iter<'a, A, D>
        where
            D: Dimension,
        {
            type Item = $array<'a, A, D>;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                self.iter
                    .next_with_index()
                    .map(|(index, ptr)| self.get_subview(index, ptr))
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.iter.size_hint()
            }
        }

        impl<'a, A, D> DoubleEndedIterator for $iter<'a, A, D>
        where
            D: Dimension,
        {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                self.iter
                    .next_back_with_index()
                    .map(|(index, ptr)| self.get_subview(index, ptr))
            }
        }

        impl<'a, A, D> ExactSizeIterator for $iter<'a, A, D> where D: Dimension {}
    };
}

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
pub struct AxisChunksIterMut<'a, A, D> {
    iter: AxisIterCore<A, D>,
    partial_chunk_index: usize,
    partial_chunk_dim: D,
    life: PhantomData<&'a mut A>,
}

impl<'a, A, D: Dimension> AxisChunksIterMut<'a, A, D> {
    pub(crate) fn new(v: ArrayViewMut<'a, A, D>, axis: Axis, size: usize) -> Self {
        let (iter, partial_chunk_index, partial_chunk_dim) =
            chunk_iter_parts(v.into_view(), axis, size);
        AxisChunksIterMut {
            iter,
            partial_chunk_index,
            partial_chunk_dim,
            life: PhantomData,
        }
    }
}

chunk_iter_impl!(AxisChunksIter, ArrayView);
chunk_iter_impl!(AxisChunksIterMut, ArrayViewMut);

// send_sync_read_only!(Iter);
send_sync_read_only!(IndexedIter);
send_sync_read_only!(LanesIter);
send_sync_read_only!(AxisIter);
send_sync_read_only!(AxisChunksIter);
send_sync_read_only!(Iter);

// send_sync_read_write!(IterMut);
send_sync_read_write!(IndexedIterMut);
send_sync_read_write!(LanesIterMut);
send_sync_read_write!(AxisIterMut);
send_sync_read_write!(AxisChunksIterMut);
send_sync_read_write!(IterMut);

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
unsafe impl<'a, A, D: Dimension> TrustedIterator for Iter<'a, A, D> {}
unsafe impl<'a, A, D: Dimension> TrustedIterator for IterMut<'a, A, D> {}
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
