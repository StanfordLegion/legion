use std::io::Write;

use crate::serialize::Record;

use crate::spy::serialize::Record as SpyRecord;

pub fn dump_record(records: &[Record]) -> std::io::Result<()> {
    let mut stdout = std::io::stdout().lock();
    for record in records {
        serde_json::to_writer(&mut stdout, record).map_err(|e| {
            let e: std::io::Error = e.into();
            e
        })?;
        writeln!(&mut stdout)?;
    }
    Ok(())
}

pub fn dump_spy_record(records: &[SpyRecord]) -> std::io::Result<()> {
    let mut stdout = std::io::stdout().lock();
    for record in records {
        serde_json::to_writer(&mut stdout, record).map_err(|e| {
            let e: std::io::Error = e.into();
            e
        })?;
        writeln!(&mut stdout)?;
    }
    Ok(())
}
