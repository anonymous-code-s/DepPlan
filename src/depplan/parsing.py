"""Parse question AMRs in timed batches of four by default."""
import math
import time


def parse_questions(rows, parser, structure_source, *, batch_size=4, synchronize=lambda: None,
                    decode=None, clock=time.perf_counter):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('AMR batch size must be a positive integer')
    if decode is None:
        import penman
        decode = penman.decode
    output = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        synchronize()
        started = clock()
        parsed, batch_error = None, None
        try:
            parsed = parser.parse_sents([row['question'] for row in batch])
            if len(parsed) != len(batch):
                raise ValueError('Parser output count differs from input batch')
        except Exception as exc:
            batch_error = type(exc).__name__
        records = []
        for i, row in enumerate(batch):
            record = {k: v for k, v in row.items()
                      if k not in {'amr_triples', 'raw_amr', 'amr_root', 'error_type'}}
            record.update(structure_source=structure_source, status='graph_failed')
            try:
                if batch_error:
                    record['error_type'] = batch_error
                else:
                    graph = decode(parsed[i])
                    if not graph.instances():
                        raise ValueError('No AMR concepts')
                    record.update(status='success', raw_amr=parsed[i], amr_root=graph.top,
                                  amr_triples=[list(t) for t in graph.triples])
            except Exception as exc:
                record['error_type'] = type(exc).__name__
            records.append(record)
        synchronize()
        elapsed = clock() - started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid AMR batch timing')
        # Amortize the measured batch cost over every input, including failures.
        for record in records:
            record.update(amr_seconds=elapsed / len(batch), amr_batch_seconds=elapsed,
                          amr_batch_id=start // batch_size, amr_batch_size=len(batch),
                          amr_configured_batch_size=batch_size,
                          amr_time_scope='batch parsing and graph decoding, amortized over batch inputs')
        output.extend(records)
    return output
