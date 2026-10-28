"""
NSE F&O Bhavcopy Downloader - 2020 to 2026 (CORRECTED VERSION)
Handles BOTH formats:
- OLD format: January 2020 to July 7, 2024
- NEW UDiFF format: July 8, 2024 onwards

This fixes the empty folder issue!
"""

import os
import requests
from datetime import datetime, timedelta
import pandas as pd
import time

# Optional: Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("💡 Tip: Install tqdm for progress bar: pip install tqdm\n")

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = r"C:\Users\pramu\Desktop\code downloaded data"

# Date range: January 2020 to January 2026
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2026, 1, 31)

# CRITICAL: UDiFF format started on July 8, 2024
UDIFF_START_DATE = datetime(2024, 7, 8)

# OLD format URL (pre-July 8, 2024)
OLD_URL_TEMPLATE = "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{year}/{month}/fo{date}bhav.csv.zip"

# NEW UDiFF format URL (July 8, 2024 onwards)
UDIFF_URL_TEMPLATE = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive"
}

# Comprehensive NSE holidays for 2020-2026
NSE_HOLIDAYS = {
    # 2020
    datetime(2020, 1, 26), datetime(2020, 2, 21), datetime(2020, 3, 10),
    datetime(2020, 3, 25), datetime(2020, 4, 2), datetime(2020, 4, 6),
    datetime(2020, 4, 10), datetime(2020, 5, 1), datetime(2020, 5, 25),
    datetime(2020, 8, 3), datetime(2020, 8, 12), datetime(2020, 8, 15),
    datetime(2020, 10, 2), datetime(2020, 10, 25), datetime(2020, 11, 14),
    datetime(2020, 11, 16), datetime(2020, 11, 30), datetime(2020, 12, 25),
    
    # 2021
    datetime(2021, 1, 26), datetime(2021, 3, 11), datetime(2021, 3, 29),
    datetime(2021, 4, 2), datetime(2021, 4, 14), datetime(2021, 4, 21),
    datetime(2021, 4, 25), datetime(2021, 5, 13), datetime(2021, 7, 21),
    datetime(2021, 8, 19), datetime(2021, 9, 10), datetime(2021, 10, 15),
    datetime(2021, 11, 4), datetime(2021, 11, 5), datetime(2021, 11, 19),
    
    # 2022
    datetime(2022, 1, 26), datetime(2022, 3, 1), datetime(2022, 3, 18),
    datetime(2022, 4, 10), datetime(2022, 4, 14), datetime(2022, 4, 15),
    datetime(2022, 5, 3), datetime(2022, 7, 10), datetime(2022, 8, 9),
    datetime(2022, 8, 15), datetime(2022, 8, 31), datetime(2022, 10, 5),
    datetime(2022, 10, 24), datetime(2022, 10, 26), datetime(2022, 11, 8),
    
    # 2023
    datetime(2023, 1, 26), datetime(2023, 2, 18), datetime(2023, 3, 7),
    datetime(2023, 3, 30), datetime(2023, 4, 4), datetime(2023, 4, 7),
    datetime(2023, 4, 14), datetime(2023, 4, 22), datetime(2023, 5, 1),
    datetime(2023, 6, 29), datetime(2023, 7, 29), datetime(2023, 8, 15),
    datetime(2023, 9, 19), datetime(2023, 10, 2), datetime(2023, 10, 24),
    datetime(2023, 11, 12), datetime(2023, 11, 13), datetime(2023, 11, 14),
    datetime(2023, 11, 27), datetime(2023, 12, 25),
    
    # 2024
    datetime(2024, 1, 22), datetime(2024, 1, 26), datetime(2024, 3, 8),
    datetime(2024, 3, 25), datetime(2024, 3, 29), datetime(2024, 4, 11),
    datetime(2024, 4, 17), datetime(2024, 4, 21), datetime(2024, 5, 1),
    datetime(2024, 5, 20), datetime(2024, 6, 17), datetime(2024, 7, 17),
    datetime(2024, 8, 15), datetime(2024, 9, 7), datetime(2024, 10, 2),
    datetime(2024, 10, 12), datetime(2024, 11, 1), datetime(2024, 11, 15),
    datetime(2024, 11, 20), datetime(2024, 12, 25),
    
    # 2025
    datetime(2025, 1, 26), datetime(2025, 2, 26), datetime(2025, 3, 14),
    datetime(2025, 3, 31), datetime(2025, 4, 6), datetime(2025, 4, 10),
    datetime(2025, 4, 14), datetime(2025, 4, 18), datetime(2025, 5, 1),
    datetime(2025, 6, 7), datetime(2025, 7, 6), datetime(2025, 8, 15),
    datetime(2025, 8, 27), datetime(2025, 10, 2), datetime(2025, 10, 21),
    datetime(2025, 10, 22), datetime(2025, 11, 5), datetime(2025, 12, 25),
    
    # 2026
    datetime(2026, 1, 26),
}

# ==========================
# HELPER FUNCTIONS
# ==========================
def month_folder(date):
    """Create folder name in MM-YYYY format"""
    return f"{date.month:02d}-{date.year}"

def file_name(date):
    """Create filename for the downloaded zip"""
    return f"FO_BHAV_{date.strftime('%Y%m%d')}.zip"

def is_weekend(date):
    """Check if date is a weekend"""
    return date.weekday() >= 5

def is_public_holiday(date):
    """Check if date is a known NSE public holiday"""
    return date in NSE_HOLIDAYS

def get_holiday_type(date):
    """Get the type of holiday"""
    if is_weekend(date):
        return "Saturday" if date.weekday() == 5 else "Sunday"
    elif is_public_holiday(date):
        return "Public Holiday"
    else:
        return "Trading Day"

def get_download_url(date):
    """Get the correct URL based on date (old or new format)"""
    if date >= UDIFF_START_DATE:
        # NEW UDiFF format (from July 8, 2024)
        return UDIFF_URL_TEMPLATE.format(date=date.strftime("%Y%m%d"))
    else:
        # OLD format (before July 8, 2024)
        year = date.strftime("%Y")
        month = date.strftime("%b").upper()  # JAN, FEB, etc.
        date_str = date.strftime("%d%b%Y").upper()  # 01JAN2020
        return OLD_URL_TEMPLATE.format(year=year, month=month, date=date_str)

def get_existing_files(base_dir):
    """Get set of already downloaded dates"""
    existing = set()
    if not os.path.exists(base_dir):
        return existing
    
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.startswith('FO_BHAV_') and file.endswith('.zip'):
                    date_str = file[8:16]  # Gets '20250115'
                    try:
                        date = datetime.strptime(date_str, '%Y%m%d')
                        existing.add(date.date())
                    except ValueError:
                        pass
    return existing

# ==========================
# MAIN DOWNLOAD LOGIC
# ==========================
def main():
    # Create base directory
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # Get already downloaded files
    existing_files = get_existing_files(BASE_DIR)
    if existing_files:
        print(f"📁 Found {len(existing_files)} already downloaded files\n")
    
    # Track statistics
    stats = {
        'downloaded': 0,
        'skipped_existing': 0,
        'skipped_weekend': 0,
        'skipped_holiday': 0,
        'missing': 0,
        'errors': 0,
        'old_format': 0,
        'new_format': 0
    }
    
    missing_dates = []
    all_holidays = []
    
    # Create session
    session = requests.Session()
    session.headers.update(HEADERS)
    
    # Initial request to NSE homepage to set cookies
    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(2)
    except:
        pass
    
    print(f"🚀 NSE F&O Bhavcopy Downloader")
    print(f"{'=' * 70}")
    print(f"📅 Date Range: {START_DATE.date()} to {END_DATE.date()}")
    print(f"📋 Format Info:")
    print(f"   • OLD format: {START_DATE.date()} to {(UDIFF_START_DATE - timedelta(days=1)).date()}")
    print(f"   • NEW UDiFF:  {UDIFF_START_DATE.date()} to {END_DATE.date()}")
    print(f"{'=' * 70}\n")
    
    # Calculate total days
    total_days = (END_DATE - START_DATE).days + 1
    current_date = START_DATE
    
    # Progress bar
    if HAS_TQDM:
        pbar = tqdm(total=total_days, desc="Progress", unit="day")
    
    while current_date <= END_DATE:
        date_str_display = current_date.strftime('%Y-%m-%d')
        holiday_type = get_holiday_type(current_date)
        
        # Log holidays
        if holiday_type != "Trading Day":
            all_holidays.append({
                'date': date_str_display,
                'day_of_week': current_date.strftime('%A'),
                'holiday_type': holiday_type
            })
        
        # Check if already downloaded
        if current_date.date() in existing_files:
            stats['skipped_existing'] += 1
            if HAS_TQDM:
                pbar.update(1)
            current_date += timedelta(days=1)
            continue
        
        # Skip weekends
        if is_weekend(current_date):
            stats['skipped_weekend'] += 1
            if HAS_TQDM:
                pbar.update(1)
            current_date += timedelta(days=1)
            continue
        
        # Skip holidays
        if is_public_holiday(current_date):
            stats['skipped_holiday'] += 1
            if HAS_TQDM:
                pbar.update(1)
            current_date += timedelta(days=1)
            continue
        
        try:
            # Create month folder
            folder_path = os.path.join(BASE_DIR, month_folder(current_date))
            os.makedirs(folder_path, exist_ok=True)
            
            # Get URL based on date
            url = get_download_url(current_date)
            
            # Download file
            response = session.get(url, timeout=30)
            
            if response.status_code == 200 and len(response.content) > 100:
                # Verify it's a ZIP file
                if response.content[:2] == b'PK' or 'zip' in response.headers.get('Content-Type', '').lower():
                    # Save file
                    file_path = os.path.join(folder_path, file_name(current_date))
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    
                    stats['downloaded'] += 1
                    
                    # Track format type
                    if current_date >= UDIFF_START_DATE:
                        stats['new_format'] += 1
                    else:
                        stats['old_format'] += 1
                    
                    if not HAS_TQDM:
                        size_kb = len(response.content) / 1024
                        format_type = "UDiFF" if current_date >= UDIFF_START_DATE else "OLD"
                        print(f"✅ {date_str_display} - {format_type} ({size_kb:.1f} KB)")
                    
                    if HAS_TQDM:
                        pbar.set_postfix({
                            '✅': stats['downloaded'],
                            'OLD': stats['old_format'],
                            'NEW': stats['new_format']
                        })
                else:
                    stats['missing'] += 1
                    missing_dates.append(date_str_display)
                    if not HAS_TQDM:
                        print(f"❌ {date_str_display} - Invalid file format")
            else:
                stats['missing'] += 1
                missing_dates.append(date_str_display)
                if not HAS_TQDM:
                    print(f"❌ {date_str_display} - HTTP {response.status_code}")
        
        except Exception as e:
            stats['errors'] += 1
            missing_dates.append(date_str_display)
            if not HAS_TQDM:
                print(f"⚠️  {date_str_display} - Error: {str(e)[:40]}")
        
        if HAS_TQDM:
            pbar.update(1)
        
        current_date += timedelta(days=1)
        time.sleep(0.5)  # Respectful delay
    
    if HAS_TQDM:
        pbar.close()
    
    # ==========================
    # SAVE LOGS
    # ==========================
    if missing_dates:
        df = pd.DataFrame(missing_dates, columns=["missing_date"])
        csv_path = os.path.join(BASE_DIR, "missing_dates.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n📄 Missing dates: {csv_path}")
    
    if all_holidays:
        df_holidays = pd.DataFrame(all_holidays)
        csv_path = os.path.join(BASE_DIR, "all_holidays.csv")
        df_holidays.to_csv(csv_path, index=False)
        print(f"📄 All holidays: {csv_path}")
        
        holiday_summary = df_holidays.groupby('holiday_type').size().reset_index(name='count')
        summary_path = os.path.join(BASE_DIR, "holiday_summary.csv")
        holiday_summary.to_csv(summary_path, index=False)
        print(f"📄 Holiday summary: {summary_path}")
    
    # ==========================
    # SUMMARY
    # ==========================
    print(f"\n{'=' * 70}")
    print(f"📊 DOWNLOAD SUMMARY")
    print(f"{'=' * 70}")
    print(f"📅 Period: {START_DATE.date()} to {END_DATE.date()} ({total_days:,} days)")
    print(f"{'=' * 70}")
    print(f"✅ Downloaded:           {stats['downloaded']:,} files")
    print(f"   • OLD format:         {stats['old_format']:,} files (pre-July 2024)")
    print(f"   • NEW UDiFF format:   {stats['new_format']:,} files (July 2024+)")
    print(f"⏭️  Already existed:      {stats['skipped_existing']:,} files")
    print(f"📅 Saturdays:            {len([h for h in all_holidays if h['holiday_type'] == 'Saturday']):,}")
    print(f"📅 Sundays:              {len([h for h in all_holidays if h['holiday_type'] == 'Sunday']):,}")
    print(f"🏖️  Public holidays:      {stats['skipped_holiday']:,}")
    print(f"❌ Missing/unavailable:  {stats['missing']:,}")
    print(f"⚠️  Errors:               {stats['errors']:,}")
    print(f"{'=' * 70}")
    print(f"📁 Files saved to: {BASE_DIR}")
    
    total_trading = stats['downloaded'] + stats['missing'] + stats['errors']
    if total_trading > 0:
        success_rate = (stats['downloaded'] / total_trading) * 100
        print(f"📈 Success rate: {success_rate:.1f}%")
    
    print(f"\n🎯 Done!")

if __name__ == "__main__":
    main()