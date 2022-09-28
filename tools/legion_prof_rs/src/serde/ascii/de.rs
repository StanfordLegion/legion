use std::fmt;

use nom::{
    bytes::complete::tag,
    character::complete::{
        alphanumeric1, i16, i32, i64, i8, not_line_ending, space0, u16, u32, u64, u8,
    },
    combinator::{eof, opt},
};

use serde::de::{
    self, DeserializeSeed, EnumAccess, IntoDeserializer, MapAccess, SeqAccess, VariantAccess,
    Visitor,
};
use serde::{self, forward_to_deserialize_any, Deserialize};

use super::error::{Error, Result};

pub struct Deserializer<'de> {
    input: &'de str,
}

impl<'de> Deserializer<'de> {
    pub fn from_str(input: &'de str) -> Self {
        Deserializer { input }
    }
}

pub fn from_str<'a, T>(s: &'a str) -> Result<T>
where
    T: Deserialize<'a>,
{
    let mut deserializer = Deserializer::from_str(s);
    let t = T::deserialize(&mut deserializer)?;
    if deserializer.input.is_empty() {
        Ok(t)
    } else {
        Err(Error::TrailingCharacters(s.to_owned()))
    }
}

impl<'de> Deserializer<'de> {
    fn parse<O, P>(&mut self, mut parser: P) -> Result<O>
    where
        P: FnMut(&'de str) -> nom::IResult<&'de str, O, nom::error::Error<&'de str>>,
    {
        let input = self.input;
        let (input, _) = space0(input)?;
        let (input, value) = parser(input)?;
        self.input = input;
        Ok(value)
    }

    fn end_of_line(&mut self) -> Result<bool> {
        let input = self.input;
        let (input, end) = opt(eof)(input)?;
        self.input = input;
        Ok(end.is_some())
    }
}

impl<'de, 'a> de::Deserializer<'de> for &'a mut Deserializer<'de> {
    type Error = Error;

    // We don't actually implement deserialize_any. This is just a way
    // to shorten the code so that we don't have a bunch of useless
    // unimplemented!() methods sitting around.
    forward_to_deserialize_any! {
        bytes byte_buf char f32 f64 ignored_any map option unit unit_struct
    }

    fn deserialize_any<V>(self, _visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        unimplemented!()
    }

    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_bool(self.parse(i8)? != 0)
    }

    fn deserialize_i8<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i8(self.parse(i8)?)
    }

    fn deserialize_i16<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i16(self.parse(i16)?)
    }

    fn deserialize_i32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i32(self.parse(i32)?)
    }

    fn deserialize_i64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(self.parse(i64)?)
    }

    fn deserialize_u8<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u8(self.parse(u8)?)
    }

    fn deserialize_u16<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u16(self.parse(u16)?)
    }

    fn deserialize_u32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u32(self.parse(u32)?)
    }

    fn deserialize_u64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u64(self.parse(u64)?)
    }

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_borrowed_str(self.parse(not_line_ending)?)
    }

    fn deserialize_string<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_str(visitor)
    }

    fn deserialize_newtype_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_seq(Sequence::new(self))
    }

    fn deserialize_tuple<V>(self, _len: usize, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_seq(visitor)
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_seq(visitor)
    }

    fn deserialize_enum<V>(
        self,
        _name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_enum(Enum::new(self, variants))
    }

    // Hack: I'm abusing this method to provide a way to parse hex digits. You
    // can't use strings for this purpose because they consume the entire rest
    // of the line.
    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_borrowed_str(self.parse(alphanumeric1)?)
    }
}

struct Sequence<'a, 'de: 'a> {
    de: &'a mut Deserializer<'de>,
}

impl<'a, 'de> Sequence<'a, 'de> {
    fn new(de: &'a mut Deserializer<'de>) -> Self {
        Sequence { de }
    }
}

impl<'de, 'a> SeqAccess<'de> for Sequence<'a, 'de> {
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>>
    where
        T: DeserializeSeed<'de>,
    {
        if self.de.end_of_line()? {
            return Ok(None);
        }
        seed.deserialize(&mut *self.de).map(Some)
    }
}

impl<'de, 'a> MapAccess<'de> for Sequence<'a, 'de> {
    type Error = Error;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>>
    where
        K: DeserializeSeed<'de>,
    {
        if self.de.end_of_line()? {
            return Ok(None);
        }
        seed.deserialize(&mut *self.de).map(Some)
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value>
    where
        V: DeserializeSeed<'de>,
    {
        seed.deserialize(&mut *self.de)
    }
}

struct Enum<'a, 'de: 'a> {
    de: &'a mut Deserializer<'de>,
    variants: &'static [&'static str],
}

impl<'a, 'de> Enum<'a, 'de> {
    fn new(de: &'a mut Deserializer<'de>, variants: &'static [&'static str]) -> Self {
        Enum { de, variants }
    }
}

impl<'de, 'a> EnumAccess<'de> for Enum<'a, 'de> {
    type Error = Error;
    type Variant = Self;

    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self::Variant)>
    where
        V: DeserializeSeed<'de>,
    {
        for variant in self.variants {
            if self.de.parse(opt(tag(*variant)))?.is_some() {
                return Ok((
                    seed.deserialize(<&str as IntoDeserializer<'de, Error>>::into_deserializer(
                        *variant,
                    ))?,
                    self,
                ));
            }
        }
        Err(Error::ExpectedEnum(self.de.input.to_owned()))
    }
}

impl<'de, 'a> VariantAccess<'de> for Enum<'a, 'de> {
    type Error = Error;

    fn unit_variant(self) -> Result<()> {
        Ok(())
    }

    fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value>
    where
        T: DeserializeSeed<'de>,
    {
        seed.deserialize(self.de)
    }

    fn tuple_variant<V>(self, _len: usize, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        de::Deserializer::deserialize_seq(self.de, visitor)
    }

    fn struct_variant<V>(self, _fields: &'static [&'static str], visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        de::Deserializer::deserialize_seq(self.de, visitor)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct HexU64(pub u64);

impl<'de> Deserialize<'de> for HexU64 {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct HexVisitor {}
        impl<'de> Visitor<'de> for HexVisitor {
            type Value = HexU64;
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a hexidecimal number")
            }

            fn visit_str<E>(self, v: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                u64::from_str_radix(v, 16)
                    .map(HexU64)
                    .map_err(serde::de::Error::custom)
            }
        }

        deserializer.deserialize_identifier(HexVisitor {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i8() {
        let a1 = "-123";
        let a2 = " -123";
        let b = -123i8;
        assert_eq!(b, from_str::<i8>(a1).unwrap());
        assert_eq!(b, from_str::<i8>(a2).unwrap());
    }

    #[test]
    fn test_i16() {
        let a1 = "-12345";
        let a2 = " -12345";
        let b = -12345i16;
        assert_eq!(b, from_str::<i16>(a1).unwrap());
        assert_eq!(b, from_str::<i16>(a2).unwrap());
    }

    #[test]
    fn test_i32() {
        let a1 = "-1234567";
        let a2 = " -1234567";
        let b = -1234567i32;
        assert_eq!(b, from_str::<i32>(a1).unwrap());
        assert_eq!(b, from_str::<i32>(a2).unwrap());
    }

    #[test]
    fn test_i64() {
        let a1 = "-1234567890";
        let a2 = " -1234567890";
        let b = -1234567890i64;
        assert_eq!(b, from_str::<i64>(a1).unwrap());
        assert_eq!(b, from_str::<i64>(a2).unwrap());
    }

    #[test]
    fn test_u8() {
        let a1 = "123";
        let a2 = " 123";
        let b = 123u8;
        assert_eq!(b, from_str::<u8>(a1).unwrap());
        assert_eq!(b, from_str::<u8>(a2).unwrap());
    }

    #[test]
    fn test_u16() {
        let a1 = "12345";
        let a2 = " 12345";
        let b = 12345u16;
        assert_eq!(b, from_str::<u16>(a1).unwrap());
        assert_eq!(b, from_str::<u16>(a2).unwrap());
    }

    #[test]
    fn test_u32() {
        let a1 = "1234567";
        let a2 = " 1234567";
        let b = 1234567u32;
        assert_eq!(b, from_str::<u32>(a1).unwrap());
        assert_eq!(b, from_str::<u32>(a2).unwrap());
    }

    #[test]
    fn test_u64() {
        let a1 = "1234567890";
        let a2 = " 1234567890";
        let b = 1234567890u64;
        assert_eq!(b, from_str::<u64>(a1).unwrap());
        assert_eq!(b, from_str::<u64>(a2).unwrap());
    }

    #[test]
    fn test_hex_u64() {
        let a1 = "1234abcd";
        let a2 = " 1234abcd";
        let b = HexU64(0x1234abcd);
        assert_eq!(b, from_str(a1).unwrap());
        assert_eq!(b, from_str(a2).unwrap());
    }

    #[test]
    fn test_string() {
        let a1 = "1234 asdf qwer";
        let a2 = " 1234 asdf qwer";
        let b = "1234 asdf qwer";
        assert_eq!(b, from_str::<String>(a1).unwrap());
        assert_eq!(b, from_str::<String>(a2).unwrap());
    }

    #[test]
    fn test_empty_string() {
        let a1 = "";
        let a2 = " ";
        let b = "";
        assert_eq!(b, from_str::<String>(a1).unwrap());
        assert_eq!(b, from_str::<String>(a2).unwrap());
    }

    #[test]
    fn test_vec() {
        let a1 = "1 2 3";
        let a2 = " 1 2 3";
        let b = vec![1, 2, 3];
        assert_eq!(b, from_str::<Vec<i32>>(a1).unwrap());
        assert_eq!(b, from_str::<Vec<i32>>(a2).unwrap());
    }

    #[test]
    fn test_tuple() {
        let a1 = "-1 12345 98765432";
        let a2 = " -1 12345 98765432";
        let b = (-1i8, 12345i16, 98765432u32);
        assert_eq!(b, from_str(a1).unwrap());
        assert_eq!(b, from_str(a2).unwrap());
    }

    #[test]
    fn test_tuple_struct() {
        #[derive(Deserialize, PartialEq, Debug)]
        struct S(i8, i16, u32);

        let a1 = "-1 12345 98765432";
        let a2 = " -1 12345 98765432";
        let b = S(-1, 12345, 98765432);
        assert_eq!(b, from_str(a1).unwrap());
        assert_eq!(b, from_str(a2).unwrap());
    }

    #[test]
    fn test_struct() {
        #[derive(Deserialize, PartialEq, Debug)]
        struct S {
            x: i8,
            y: i16,
            z: u32,
        }

        let a1 = "-1 12345 98765432";
        let a2 = " -1 12345 98765432";
        let b = S {
            x: -1,
            y: 12345,
            z: 98765432,
        };
        assert_eq!(b, from_str(a1).unwrap());
        assert_eq!(b, from_str(a2).unwrap());
    }

    #[test]
    fn test_enum() {
        #[derive(Deserialize, PartialEq, Debug)]
        enum E {
            A,
            B(i32),
            C { x: i8, y: i16, z: u32 },
        }

        let a1 = "A";
        let a2 = " A";
        let b = E::A;
        assert_eq!(b, from_str(a1).unwrap());
        assert_eq!(b, from_str(a2).unwrap());

        let c1 = "B -123";
        let c2 = " B -123";
        let d = E::B(-123);
        assert_eq!(d, from_str(c1).unwrap());
        assert_eq!(d, from_str(c2).unwrap());

        let e1 = "C -1 12345 98765432";
        let e2 = " C -1 12345 98765432";
        let f = E::C {
            x: -1,
            y: 12345,
            z: 98765432,
        };
        assert_eq!(f, from_str(e1).unwrap());
        assert_eq!(f, from_str(e2).unwrap());
    }
}
