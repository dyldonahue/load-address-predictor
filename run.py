# ================================================
# Author: Dylan Donahue
# 11.24.2025
# Parser to interpret the ChampSim SPEC CPU2017 traces
# ================================================
import lzma
import struct
import sys
import glob
from pathlib import Path
from datetime import datetime
from base_predictor import BaseAddressPredictor
from custom_predictor import CustomAddressPredictor

class TraceReader:
    """Read ChampSim binary traces."""
    
    def __init__(self, trace_file):
        self.trace_file = trace_file
        self.file_handle = None
    
    def __enter__(self):
        self.file_handle = lzma.open(self.trace_file, 'rb')
        return self
    
    def __exit__(self, *args):
        if self.file_handle:
            self.file_handle.close()
    
    def read_entries(self, max_entries=None):
        """Generator that yields (ip, address) tuples for valid memory operations."""
        count = 0
        
        while True:
            # Read 16 bytes (IP + addr)
            data = self.file_handle.read(16)
            
            if len(data) < 16:
                break
            
            # Parse
            ip, address = struct.unpack('QQ', data)
            
            # Skip invalid entries
            if ip != 0 and address != 0 and address < 2**48:
                yield ip, address
                count += 1
                
                if max_entries and count >= max_entries:
                    break


def run_predictor_on_trace(trace_file, max_entries=None, verbose=True, predictor_mode='base'):
    """Run predictor on ChampSim trace"""
    
    if predictor_mode == 'custom':
        predictor = CustomAddressPredictor(
            prediction_window=100,
        )
    else:
        predictor = BaseAddressPredictor(
            prediction_window=100,
        )
    
    if verbose:
        print(f"Processing: {trace_file}")
        print(f"Max entries: {max_entries if max_entries else 'unlimited'}\n")
    
    valid_count = 0
    
    with TraceReader(trace_file) as reader:
        for ip, address in reader.read_entries(max_entries):
            predictor.access(ip, address)
            valid_count += 1
            
            if verbose and valid_count % 1000000 == 0:
                print(f"  Processed {valid_count:,} entries...", end='\r')
    
    if verbose:
        print(f"\n  Processed {valid_count:,} total entries")
    
    predictor.finalize()
    return predictor, valid_count


def save_results_to_file(trace_name, stats, valid_count, output_file):
    """Append results to output file"""
    with open(output_file, 'a') as f:
        f.write("="*80 + "\n")
        f.write(f"Trace: {trace_name}\n")
        f.write(f"Entries Processed: {valid_count:,}\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Accesses: {stats['total_accesses']:,}\n")
        f.write(f"Total Predictions: {stats['total_predictions']:,}\n")
        f.write(f"  Useful:   {stats['useful_predictions']:,}\n")
        f.write(f"  Useless:  {stats['useless_predictions']:,}\n")
        f.write(f"  Pending:  {stats['pending_predictions']:,}\n")
        f.write(f"Accuracy: {stats['accuracy']:.1%}\n")
        f.write(f"Tracked PCs: {stats['tracked_pcs']}\n")
        if stats['total_accesses'] > 0:
            f.write(f"Predictions/Access: {stats['total_predictions']/stats['total_accesses']:.2f}\n")
        f.write("\n")


def run_batch(trace_pattern='./traces/*.xz', max_entries=100_000_000, 
              output_file='results.txt', predictor_mode='base'):
    """Run predictor on all traces matching pattern"""
    
    traces = sorted(glob.glob(trace_pattern))
    
    if not traces:
        print(f"No traces found matching: {trace_pattern}")
        return
    
    print(f"Found {len(traces)} traces")
    print(f"Output file: {output_file}")
    print(f"Max entries per trace: {max_entries:,}\n")
    
    # Clear/create output file with header
    with open(output_file, 'w') as f:
        f.write(f"ChampSim Trace Evaluation Results\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Max entries per trace: {max_entries:,}\n")
        f.write("="*80 + "\n\n")
    
    # Process each trace
    results = []
    
    for i, trace_file in enumerate(traces, 1):
        trace_name = Path(trace_file).name
        
        print(f"\n[{i}/{len(traces)}] {trace_name}")
        print("="*80)
        
        try:
            predictor, valid_count = run_predictor_on_trace(
                trace_file, 
                max_entries=max_entries,
                verbose=True,
                predictor_mode=predictor_mode,
            )
            
            stats = predictor.get_stats()
            
            # Print to console
            predictor.print_stats()
            
            # Save to file
            save_results_to_file(trace_name, stats, valid_count, output_file)
            
            # Store for summary
            results.append({
                'name': trace_name,
                'accuracy': stats['accuracy'],
                'predictions': stats['total_predictions'],
                'useful': stats['useful_predictions']
            })
            
        except Exception as e:
            error_msg = f"ERROR processing {trace_name}: {e}"
            print(f"\n{error_msg}\n")
            
            with open(output_file, 'a') as f:
                f.write(f"Trace: {trace_name}\n")
                f.write(f"{error_msg}\n\n")
    
    # Write summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    with open(output_file, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        
        for result in results:
            line = f"{result['name']:50s} | Accuracy: {result['accuracy']:6.1%} | Predictions: {result['predictions']:10,}"
            print(line)
            f.write(line + "\n")
        
        # Overall average
        if results:
            avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
            total_predictions = sum(r['predictions'] for r in results)
            total_useful = sum(r['useful'] for r in results)
            overall_accuracy = total_useful / total_predictions if total_predictions > 0 else 0
            
            summary = f"\nAverage Accuracy: {avg_accuracy:.1%}\nOverall Accuracy: {overall_accuracy:.1%}"
            print(summary)
            f.write(summary + "\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run load address predictor on ChampSim traces')
    parser.add_argument('trace', nargs='?', default=None,
                       help='Specific trace file to run (if not provided, runs all in ./traces/)')
    parser.add_argument('-n', '--max-entries', type=int, default=100_000_000,
                       help='Maximum entries to process per trace (default: 100M)')
    parser.add_argument('-o', '--output', type=str, default='results.txt',
                       help='Output file for results (default: results.txt)')
    parser.add_argument('-p', '--pattern', type=str, default='./traces/*.xz',
                       help='Pattern for batch mode (default: ./traces/*.xz)')
    parser.add_argument('-m', '--mode', type=str, choices=['base', 'custom'], default='base',)
    
    args = parser.parse_args()
    
    if args.trace:
        # Single trace mode
        print(f"Running on single trace: {args.trace}")
        predictor, valid_count = run_predictor_on_trace(
            args.trace, 
            max_entries=args.max_entries,
            verbose=True, 
            predictor_mode=args.mode,
        )
        predictor.print_stats()
        
        # Save to file
        stats = predictor.get_stats()
        trace_name = Path(args.trace).name
        save_results_to_file(trace_name, stats, valid_count, args.output)
        print(f"\nResults saved to: {args.output}")
        
    else:
        # Batch mode
        print("Running in batch mode on all traces")
        run_batch(
            trace_pattern=args.pattern,
            max_entries=args.max_entries,
            output_file=args.output,
            predictor_mode=args.mode,
        )