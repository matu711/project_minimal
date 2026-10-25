#!/usr/bin/env python3
"""Copy an unchanged message window from a real MCAP for reproducible benchmarks."""
import argparse
from pathlib import Path


def main():
    from mcap.reader import make_reader
    from mcap.writer import Writer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--offset', type=float, default=2)
    parser.add_argument('--seconds', type=float, default=10)
    args = parser.parse_args()
    if args.output.exists() or args.source.resolve() == args.output.resolve():
        parser.error('Choose a new output path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.source.open('rb') as source, args.output.open('xb') as target:
        reader = make_reader(source, validate_crcs=True)
        summary = reader.get_summary()
        begin = summary.statistics.message_start_time + round(args.offset*1e9)
        end = begin + round(args.seconds*1e9)
        writer = Writer(target)
        writer.start(profile='ros1')
        schemas, channels, count = {}, {}, 0
        for schema, channel, message in reader.iter_messages(start_time=begin,end_time=end):
            if schema and schema.id not in schemas:
                schemas[schema.id] = writer.register_schema(schema.name,schema.encoding,schema.data)
            if channel.id not in channels:
                channels[channel.id] = writer.register_channel(channel.topic,channel.message_encoding,
                    schemas.get(channel.schema_id,0),channel.metadata)
            writer.add_message(channels[channel.id],message.log_time,message.data,message.publish_time,message.sequence)
            count += 1
        writer.finish()
    print(f'Copied {count} original messages without decoding or changing payloads: {args.output}')


if __name__ == '__main__':
    main()
