pub trait Postincrement {
    fn postincrement(&mut self) -> Self;
}

impl Postincrement for u32 {
    fn postincrement(&mut self) -> Self {
        let value = *self;
        *self += 1;
        value
    }
}
