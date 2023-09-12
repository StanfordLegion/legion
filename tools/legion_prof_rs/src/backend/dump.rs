use crate::serialize::Record;

use crate::spy::serialize::Record as SpyRecord;

pub fn dump_record(records: &[Record]) -> serde_json::Result<()> {
    for record in records {
        serde_json::to_writer(std::io::stdout(), record)?;
        println!();
    }
    Ok(())
}

pub fn dump_spy_record(records: &[SpyRecord]) -> serde_json::Result<()> {
    for record in records {
        serde_json::to_writer(std::io::stdout(), record)?;
        println!();
    }
    Ok(())
}
