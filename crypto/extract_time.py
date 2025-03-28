import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import os
import time
import traceback  # To format exceptions for display

# --- Copy your CCXT fetching function here ---
# (Modified slightly to accept a status callback)
import ccxt
import pandas as pd
from datetime import datetime, UTC, timedelta

def fetch_full_history_ccxt_with_search(exchange_id='huobi',
                                        symbol='LTC/USDT',
                                        timeframe='1d',
                                        start_date_str='2015-01-01',
                                        search_increment_days=30,
                                        limit=1000,
                                        status_update_callback=print, # Function to call for status updates
                                        stop_event=None): # Event to signal stopping
    """
    Fetches complete historical OHLCV data using CCXT, automatically searching
    forward from start_date_str to find the actual beginning of the data.
    Includes status callback and stop event for GUI integration.
    """
    exchange_instance = None
    try:
        # Initialize exchange
        try:
            exchange_instance = ccxt.huobi()
            status_update_callback(f"Initialized {exchange_instance.name} (using ccxt.huobi).")
        except AttributeError:
             status_update_callback("ccxt.huobi not found, trying ccxt.htx...")
             try:
                 exchange_instance = ccxt.htx()
                 status_update_callback(f"Initialized {exchange_instance.name} (using ccxt.htx).")
             except AttributeError:
                  status_update_callback("Error: Neither 'huobi' nor 'htx' exchange ID found in CCXT.")
                  return None # Return None on critical failure
        # Add other exchange initializations if needed
        # elif exchange_id == 'binance': ...

    except Exception as e:
        status_update_callback(f"Error initializing exchange {exchange_id}: {e}")
        return None

    # --- Check Capabilities and Market ---
    if not exchange_instance.has['fetchOHLCV']:
        status_update_callback(f"Error: Exchange '{exchange_instance.id}' does not support fetchOHLCV.")
        return None
    try:
        status_update_callback("Loading markets...")
        exchange_instance.load_markets()
        if symbol not in exchange_instance.markets:
            status_update_callback(f"Error: Symbol '{symbol}' not found on {exchange_instance.name}. Check symbol format.")
            # Consider showing available markets example
            return None
        status_update_callback("Markets loaded successfully.")
    except Exception as e:
        status_update_callback(f"Warning: Could not load/check markets: {e}. Proceeding anyway.")

    all_ohlcv = []
    try:
        since = exchange_instance.parse8601(f"{start_date_str}T00:00:00Z")
    except Exception as e:
        status_update_callback(f"Error parsing start date '{start_date_str}': {e}")
        return None

    current_time_ms = int(time.time() * 1000)
    first_batch_found = False
    timeframe_duration_ms = exchange_instance.parse_timeframe(timeframe) * 1000
    search_increment_ms = search_increment_days * 24 * 60 * 60 * 1000
    earliest_found_dt = None # Store the first found date

    status_update_callback(f"\nAttempting to find the earliest data for {symbol} on {exchange_instance.name} ({timeframe})...")
    status_update_callback(f"Starting search around {start_date_str} UTC")

    # --- Loop 1: Search Forward ---
    while not first_batch_found and since < current_time_ms:
        if stop_event and stop_event.is_set():
            status_update_callback("Stop requested during search.")
            return None

        fetch_since_dt = datetime.fromtimestamp(since/1000, UTC)
        status_update_callback(f"Checking for data since {fetch_since_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC...")
        try:
            ohlcv = exchange_instance.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

            if ohlcv:
                status_update_callback(f"Success! Found the first batch of {len(ohlcv)} candles.")
                earliest_found_dt = datetime.fromtimestamp(ohlcv[0][0]/1000, UTC)
                status_update_callback(f"Earliest data point in this batch: {earliest_found_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                all_ohlcv.extend(ohlcv)
                first_batch_found = True
                last_candle_timestamp = ohlcv[-1][0]
                since = last_candle_timestamp + timeframe_duration_ms
            else:
                status_update_callback(f"No data found starting {fetch_since_dt.strftime('%Y-%m-%d')}. Advancing search...")
                since += search_increment_ms
                if since >= current_time_ms:
                    status_update_callback("Search start time has advanced beyond current time. Stopping search.")
                    break

            delay_seconds = max(exchange_instance.rateLimit / 1000, 0.1) # Ensure minimum delay
            status_update_callback(f"Waiting {delay_seconds:.2f} seconds...")
            # Check stop event frequently during sleep
            for _ in range(int(delay_seconds * 10)): # Check every 0.1s
                if stop_event and stop_event.is_set():
                     status_update_callback("Stop requested during wait.")
                     return None
                time.sleep(0.1)
            remaining_sleep = delay_seconds - int(delay_seconds*10)*0.1
            if remaining_sleep > 0: time.sleep(remaining_sleep)


        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            status_update_callback(f"\nNetwork error during initial search: {e}. Retrying after 10 seconds...")
            time.sleep(10)
        except ccxt.ExchangeError as e:
            status_update_callback(f"\nExchange error during initial search: {e}. Stopping search.")
            return None
        except Exception as e:
            status_update_callback(f"\nUnexpected error during initial search:\n{traceback.format_exc()}")
            return None

    if not first_batch_found:
        status_update_callback("\nCould not find any historical data after searching forward.")
        return None

    status_update_callback(f"\nFound starting point around {earliest_found_dt.strftime('%Y-%m-%d')}. Fetching remaining history...")

    # --- Loop 2: Main Pagination ---
    while since < current_time_ms:
        if stop_event and stop_event.is_set():
            status_update_callback("Stop requested during pagination.")
            return None

        fetch_since_dt = datetime.fromtimestamp(since/1000, UTC)
        status_update_callback(f"Fetching {limit} candles since {fetch_since_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC...")
        try:
            ohlcv = exchange_instance.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

            if not ohlcv:
                status_update_callback("No more data returned (likely reached current time). Fetch complete.")
                break

            all_ohlcv.extend(ohlcv)
            last_candle_timestamp = ohlcv[-1][0]
            next_since = last_candle_timestamp + timeframe_duration_ms

            if next_since == since:
                 status_update_callback("Stopping fetch: No timestamp progress.")
                 break

            since = next_since
            status_update_callback(f"Fetched {len(ohlcv)} candles up to {datetime.fromtimestamp(last_candle_timestamp / 1000, UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")

            delay_seconds = max(exchange_instance.rateLimit / 1000, 0.1) # Ensure minimum delay
            status_update_callback(f"Waiting {delay_seconds:.2f} seconds...")
            # Check stop event frequently during sleep
            for _ in range(int(delay_seconds * 10)): # Check every 0.1s
                 if stop_event and stop_event.is_set():
                     status_update_callback("Stop requested during wait.")
                     return None
                 time.sleep(0.1)
            remaining_sleep = delay_seconds - int(delay_seconds*10)*0.1
            if remaining_sleep > 0: time.sleep(remaining_sleep)

        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            status_update_callback(f"\nNetwork error during pagination: {e}. Retrying after 10 seconds...")
            time.sleep(10)
        except ccxt.ExchangeError as e:
            status_update_callback(f"\nExchange error during pagination: {e}. Stopping.")
            break
        except Exception as e:
            status_update_callback(f"\nUnexpected error during pagination:\n{traceback.format_exc()}")
            break

    if not all_ohlcv:
        status_update_callback("\nNo data collected overall.")
        return None

    # --- Process Data ---
    try:
        status_update_callback("\nProcessing all fetched data...")
        df = pd.DataFrame(all_ohlcv, columns=['timestamp_ms', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
        df.drop_duplicates(subset='timestamp_ms', inplace=True)
        df.sort_values('datetime', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        status_update_callback(f"Finished processing. Total unique records: {len(df)}")
        if not df.empty:
            status_update_callback(f"Final data range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
        return df # Return the DataFrame
    except Exception as e:
        status_update_callback(f"\nError processing DataFrame:\n{traceback.format_exc()}")
        return None


# --- GUI Application Class ---
class CryptoFetcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CCXT Crypto Data Fetcher")
        self.root.geometry("700x550") # Adjusted size

        self.fetch_thread = None
        self.stop_event = threading.Event()

        # --- Input Frame ---
        input_frame = ttk.LabelFrame(root, text="Parameters")
        input_frame.pack(padx=10, pady=10, fill="x")

        # Grid layout configuration
        input_frame.columnconfigure(1, weight=1) # Make entry column expand

        # Exchange
        ttk.Label(input_frame, text="Exchange ID:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.exchange_var = tk.StringVar(value="huobi") # Default value
        self.exchange_entry = ttk.Entry(input_frame, textvariable=self.exchange_var, width=40)
        self.exchange_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(input_frame, text="(e.g., binance, huobi, kraken)").grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Symbol
        ttk.Label(input_frame, text="Symbol:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.symbol_var = tk.StringVar(value="LTC/USDT")
        self.symbol_entry = ttk.Entry(input_frame, textvariable=self.symbol_var, width=40)
        self.symbol_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(input_frame, text="(e.g., BTC/USDT, ETH/BTC)").grid(row=1, column=2, padx=5, pady=5, sticky="w")

        # Timeframe
        ttk.Label(input_frame, text="Timeframe:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.timeframe_var = tk.StringVar(value="1d")
        timeframe_options = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        self.timeframe_combo = ttk.Combobox(input_frame, textvariable=self.timeframe_var, values=timeframe_options, width=37)
        self.timeframe_combo.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # Start Date
        ttk.Label(input_frame, text="Start Search Date:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.start_date_var = tk.StringVar(value="2017-01-01")
        self.start_date_entry = ttk.Entry(input_frame, textvariable=self.start_date_var, width=40)
        self.start_date_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(input_frame, text="(YYYY-MM-DD)").grid(row=3, column=2, padx=5, pady=5, sticky="w")

        # Output Directory
        ttk.Label(input_frame, text="Output Directory:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.output_dir_var = tk.StringVar(value=r"C:\Users\lewka\deep_learning\trading") # Default
        self.output_dir_entry = ttk.Entry(input_frame, textvariable=self.output_dir_var, width=40)
        self.output_dir_entry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        self.browse_button = ttk.Button(input_frame, text="Browse...", command=self.browse_directory)
        self.browse_button.grid(row=4, column=2, padx=5, pady=5, sticky="w")

        # --- Control Frame ---
        control_frame = ttk.Frame(root)
        control_frame.pack(padx=10, pady=5, fill="x")

        self.fetch_button = ttk.Button(control_frame, text="Fetch Data and Save", command=self.start_fetch_thread)
        self.fetch_button.pack(side="left", padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop Fetch", command=self.stop_fetch, state="disabled")
        self.stop_button.pack(side="left", padx=5)

        # --- Status Frame ---
        status_frame = ttk.LabelFrame(root, text="Status Log")
        status_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.status_text = scrolledtext.ScrolledText(status_frame, wrap=tk.WORD, height=15, state="disabled")
        self.status_text.pack(padx=5, pady=5, fill="both", expand=True)

    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory: # If user selected a directory (didn't cancel)
            self.output_dir_var.set(directory)

    def update_status(self, message):
        """Safely updates the status text area from any thread."""
        if self.root: # Check if root window still exists
            self.root.after(0, self._append_status, message) # Schedule the update

    def _append_status(self, message):
        """Internal method to append text, run by the main thread via .after()"""
        try:
            self.status_text.config(state="normal") # Enable writing
            self.status_text.insert(tk.END, str(message) + "\n")
            self.status_text.see(tk.END) # Auto-scroll
            self.status_text.config(state="disabled") # Disable writing
        except tk.TclError:
             pass # Handle case where window is closed while message is pending

    def start_fetch_thread(self):
        """Validates input and starts the data fetching in a background thread."""
        exchange_id = self.exchange_var.get().strip().lower()
        symbol = self.symbol_var.get().strip()
        timeframe = self.timeframe_var.get().strip()
        start_date_str = self.start_date_var.get().strip()
        output_dir = self.output_dir_var.get().strip()

        # Basic validation
        if not all([exchange_id, symbol, timeframe, start_date_str, output_dir]):
            self.update_status("Error: All fields are required.")
            return
        if "/" not in symbol and "-" not in symbol: # Simple check for pair format
             self.update_status(f"Warning: Symbol '{symbol}' might not be in the correct format (e.g., BTC/USDT).")
        try:
             datetime.strptime(start_date_str, '%Y-%m-%d')
        except ValueError:
             self.update_status("Error: Start Date format must be YYYY-MM-DD.")
             return
        if not os.path.isdir(output_dir):
             self.update_status(f"Error: Output directory '{output_dir}' does not exist or is not a directory.")
             return

        self.status_text.config(state="normal")
        self.status_text.delete('1.0', tk.END) # Clear previous log
        self.status_text.config(state="disabled")

        self.update_status("Starting data fetch process...")
        self.fetch_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.stop_event.clear() # Reset stop event

        # Create and start the thread
        self.fetch_thread = threading.Thread(
            target=self.run_fetch_and_save,
            args=(exchange_id, symbol, timeframe, start_date_str, output_dir),
            daemon=True # Allows program to exit even if thread is running (optional)
        )
        self.fetch_thread.start()

        # Optionally, check thread status periodically (less critical with daemon=True)
        # self.root.after(100, self.check_fetch_thread)

    def stop_fetch(self):
        if self.fetch_thread and self.fetch_thread.is_alive():
            self.update_status(">>> Sending stop signal...")
            self.stop_event.set()
            self.stop_button.config(state="disabled") # Prevent multiple clicks
            # Optionally wait a short time for thread to acknowledge, but don't block GUI
        else:
            self.update_status("No fetch process is currently running.")

    # def check_fetch_thread(self):
    #     """Checks if the thread is finished (optional alternative)."""
    #     if self.fetch_thread and not self.fetch_thread.is_alive():
    #         self.update_status("Fetch process finished.")
    #         self.fetch_button.config(state="normal")
    #         self.stop_button.config(state="disabled")
    #         self.fetch_thread = None
    #     elif self.fetch_thread and self.fetch_thread.is_alive():
    #         # Reschedule check if still running
    #         self.root.after(500, self.check_fetch_thread)


    def run_fetch_and_save(self, exchange_id, symbol, timeframe, start_date_str, output_dir):
        """The function that runs in the background thread."""
        history_df = None
        try:
            history_df = fetch_full_history_ccxt_with_search(
                exchange_id=exchange_id,
                symbol=symbol,
                timeframe=timeframe,
                start_date_str=start_date_str,
                status_update_callback=self.update_status, # Pass GUI update function
                stop_event=self.stop_event # Pass stop event
            )

            if self.stop_event.is_set():
                self.update_status("Fetch stopped by user request.")

            elif history_df is not None and not history_df.empty:
                self.update_status("\nData fetch successful. Preparing to save...")

                # --- Save to Excel ---
                safe_symbol = symbol.replace('/', '_').replace('-', '_') # Make filename safe
                excel_filename = f"{exchange_id}_{safe_symbol}_{timeframe}_full_history_ccxt.xlsx"
                excel_filepath = os.path.join(output_dir, excel_filename)

                self.update_status(f"Saving data to: {excel_filepath}")
                try:
                    # Remove timezone for Excel compatibility
                    self.update_status("Preparing datetime column for Excel...")
                    history_df['datetime'] = history_df['datetime'].dt.tz_localize(None)

                    history_df.to_excel(excel_filepath, index=False, engine='openpyxl')
                    self.update_status("Data successfully saved to Excel.")
                except Exception as e:
                    self.update_status(f"\nError saving to Excel:\n{traceback.format_exc()}")

            elif history_df is not None and history_df.empty:
                 self.update_status("\nFetch completed, but no data was returned.")
            # else: # fetch function returned None (critical error happened)
            #    self.update_status("\nFetch failed due to critical error during data retrieval or processing.")


        except Exception as e:
            # Catch any unexpected errors in this wrapper function
            self.update_status(f"\nAn unexpected error occurred in the fetch thread:\n{traceback.format_exc()}")
        finally:
            # --- Re-enable buttons on the main thread ---
            self.root.after(0, self._finalize_ui)

    def _finalize_ui(self):
        """Executed by the main thread to update UI after thread finishes/stops."""
        self.fetch_button.config(state="normal")
        self.stop_button.config(state="disabled")
        if self.stop_event.is_set():
             self.update_status("Fetch process stopped.")
        else:
             self.update_status("Fetch process finished.")
        self.fetch_thread = None # Clear thread reference
        self.stop_event.clear() # Ready for next run


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = CryptoFetcherApp(root)
    root.mainloop()
