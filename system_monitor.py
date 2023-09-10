import psutil
import csv
import time


def monitor_system(rows_to_save):
    with open("system_monitoring.csv", "w", newline="") as csvfile:
        fieldnames = [
            "Time",
            "RAM Usage (%)",
            "CPU Utilization (%)",
            "Network Bandwidth (MB/s)",
            "IO Bandwidth (MB/s)",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(rows_to_save):
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            ram_usage = psutil.virtual_memory().percent
            cpu_utilization = psutil.cpu_percent(interval=1)
            network_stats = psutil.net_io_counters()
            io_stats = psutil.disk_io_counters()

            writer.writerow(
                {
                    "Time": current_time,
                    "RAM Usage (%)": ram_usage,
                    "CPU Utilization (%)": cpu_utilization,
                    "Network Bandwidth (MB/s)": round(
                        (network_stats.bytes_sent + network_stats.bytes_recv)
                        / (1024 * 1024),
                        2,
                    ),
                    "IO Bandwidth (MB/s)": round(
                        (io_stats.read_bytes + io_stats.write_bytes) / (1024 * 1024), 2
                    ),
                }
            )
            print(_.real)

            # time.sleep(1)


if __name__ == "__main__":
    rows_to_save = int(input("Enter the number of rows to save: "))
    monitor_system(rows_to_save)

    print(f"Monitoring complete. Data saved in 'system_monitoring.csv'")
