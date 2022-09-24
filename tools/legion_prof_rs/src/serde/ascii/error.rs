use std;
use std::fmt::{self, Display};

use nom;

use serde::de;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Message(String),

    Nom(String),

    ExpectedEnum(String),

    TrailingCharacters(String),
}

impl de::Error for Error {
    fn custom<T: Display>(msg: T) -> Self {
        Error::Message(msg.to_string())
    }
}

impl<'a> From<nom::Err<nom::error::Error<&'a str>>> for Error {
    fn from(err: nom::Err<nom::error::Error<&'a str>>) -> Self {
        Error::Nom(format!("{}", err))
    }
}

impl Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Message(msg) => formatter.write_str(msg),
            _ => unimplemented!(),
        }
    }
}

impl std::error::Error for Error {}
