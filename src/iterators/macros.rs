// Send and Sync
// All the iterators are thread safe the same way the slice's iterator are

// read-only iterators use Sync => Send rules, same as `std::slice::Iter`.
macro_rules! send_sync_read_only {
    ($name:ident) => {
        unsafe impl<'a, A, D: Dimension> Send for $name<'a, A, D>
        where
            A: Sync,
            D: Send,
        {
        }
        unsafe impl<'a, A, D: Dimension> Sync for $name<'a, A, D>
        where
            A: Sync,
            D: Sync,
        {
        }
    };
}

// read-write iterators use Send => Send rules, same as `std::slice::IterMut`.
macro_rules! send_sync_read_write {
    ($name:ident) => {
        unsafe impl<'a, A, D: Dimension> Send for $name<'a, A, D>
        where
            A: Send,
            D: Send,
        {
        }
        unsafe impl<'a, A, D: Dimension> Sync for $name<'a, A, D>
        where
            A: Sync,
            D: Sync,
        {
        }
    };
}

macro_rules! send_sync_bi_array_view {
    ($name:ident,$A_trait:ident,$idx:expr) => {
        unsafe impl<'a, A, D: Dimension, DI: Dimension> Send for BaseIter<A, D, $idx, $name<'a, A, DI>>
        where
            A: $A_trait,
            D: Send,
        {
        }
        unsafe impl<'a, A, D: Dimension, DI: Dimension> Sync for BaseIter<A, D, $idx, $name<'a, A, DI>>
        where
            A: Sync,
            D: Sync,
        {
        }
    };
    ($name:ident,$A_trait:ident) => {
        send_sync_bi_array_view!($name,$A_trait,false);
        send_sync_bi_array_view!($name,$A_trait,true);
    };
}

macro_rules! impl_ndproducer {
    (
    [$($typarm:tt)*]
    [Clone => $($cloneparm:tt)*]
     $typename:ident {
         $base:ident,
         $(
             $fieldname:ident,
         )*
     }
     $fulltype:ty {
        $(
            type $atyn:ident = $atyv:ty;
        )*

        unsafe fn item(&$self_:ident, $ptr:pat) {
            $refexpr:expr
        }
    }) => {
impl<$($typarm)*> NdProducer for $fulltype {
    $(
        type $atyn = $atyv;
    )*
    type Ptr = *mut A;
    type Stride = isize;

    fn raw_dim(&self) -> D {
        self.$base.raw_dim()
    }

    fn layout(&self) -> Layout {
        self.$base.layout()
    }

    fn as_ptr(&self) -> *mut A {
        self.$base.as_ptr() as *mut _
    }

    fn contiguous_stride(&self) -> isize {
        self.$base.contiguous_stride()
    }

    unsafe fn as_ref(&$self_, $ptr: *mut A) -> Self::Item {
        $refexpr
    }

    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut A {
        self.$base.uget_ptr(i)
    }

    fn stride_of(&self, axis: Axis) -> isize {
        self.$base.stride_of(axis)
    }

    fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
        let (a, b) = self.$base.split_at(axis, index);
        ($typename {
            $base: a,
            $(
            $fieldname: self.$fieldname.clone(),
            )*
        },
        $typename {
            $base: b,
            $(
            $fieldname: self.$fieldname,
            )*
        })
    }

    private_impl!{}
}

expand_if!(@nonempty [$($cloneparm)*]
    impl<$($cloneparm)*> Clone for $fulltype {
        fn clone(&self) -> Self {
            $typename {
                $base: self.base.clone(),
                $(
                $fieldname: self.$fieldname.clone(),
                )*
            }
        }
    }
);

    }
}

macro_rules! unwrapBI {
    ($bi:expr, $variant:ident, $inner:pat => $result:expr) => {
        if let BaseIter::$variant($inner) = $bi {
            $result
        } else {
            unsafe { unreachable_unchecked() }
        }
    };
    ($bi:expr, $variant:ident) => {unwrapBI!($bi,$variant,inner=>inner)}
}