"""
Create Excel Tracker from Betting Recommendations
Generates a formatted Excel file with formulas for tracking results

Usage:
    python create_tracker.py
    python create_tracker.py --input betting_recommendations.csv --output tracker.xlsx
"""

import pandas as pd
import argparse
from datetime import datetime

def create_excel_tracker(input_file: str, output_file: str):
    """Create formatted Excel tracker with formulas."""
    
    # Load recommendations
    df = pd.read_csv(input_file)
    
    # Ensure required columns exist
    if 'actual' not in df.columns:
        df['actual'] = ''
    if 'result' not in df.columns:
        df['result'] = ''
    if 'units_bet' not in df.columns:
        df['units_bet'] = 1.0
    if 'profit' not in df.columns:
        df['profit'] = ''
    if 'date' not in df.columns:
        df['date'] = datetime.now().strftime('%Y-%m-%d')
    
    # Reorder and select columns
    cols = ['date', 'player', 'prop', 'direction', 'line', 'pred', 'edge', 
            'odds', 'strength', 'label', 'actual', 'result', 'units_bet', 'profit']
    
    # Only include columns that exist
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    # Sort by strength (descending) then edge (descending)
    if 'strength' in df.columns and 'edge' in df.columns:
        df = df.sort_values(['strength', 'edge'], ascending=[False, False])
    
    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main tracking sheet
        df.to_excel(writer, sheet_name='Bets', index=False)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Bets']
        
        # Add formulas for profit calculation (assuming data starts at row 2)
        for row in range(2, len(df) + 2):
            # Profit formula: =IF(L{row}="WIN", M{row}*IF(H{row}>0, H{row}/100, 100/ABS(H{row})), IF(L{row}="LOSS", -M{row}, ""))
            # L = result, M = units_bet, H = odds
            profit_formula = f'=IF(L{row}="WIN",M{row}*IF(H{row}>0,H{row}/100,100/ABS(H{row})),IF(L{row}="LOSS",-M{row},""))'
            worksheet.cell(row=row, column=14, value=profit_formula)
        
        # Create summary sheet
        summary_data = {
            'Metric': [
                'Total Bets',
                'Wins',
                'Losses',
                'Pending',
                'Win Rate',
                'Total Profit (Units)',
                'ROI %',
                '',
                'Strong Bets',
                'Moderate Bets', 
                'Slight Bets',
                '',
                'OVER Bets',
                'UNDER Bets'
            ],
            'Value': [
                f'=COUNTA(Bets!A2:A{len(df)+1})',
                '=COUNTIF(Bets!L:L,"WIN")',
                '=COUNTIF(Bets!L:L,"LOSS")',
                f'=COUNTA(Bets!A2:A{len(df)+1})-COUNTIF(Bets!L:L,"WIN")-COUNTIF(Bets!L:L,"LOSS")',
                '=IF(B2+B3>0,B2/(B2+B3),0)',
                '=SUM(Bets!N:N)',
                '=IF(SUM(Bets!M:M)>0,B6/SUM(Bets!M:M)*100,0)',
                '',
                '=COUNTIF(Bets!I:I,3)',
                '=COUNTIF(Bets!I:I,2)',
                '=COUNTIF(Bets!I:I,1)',
                '',
                '=COUNTIF(Bets!D:D,"OVER")',
                '=COUNTIF(Bets!D:D,"UNDER")'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Props breakdown sheet
        props_summary = df.groupby('prop').agg({
            'player': 'count',
            'edge': 'mean'
        }).reset_index()
        props_summary.columns = ['Prop', 'Bet Count', 'Avg Edge']
        props_summary.to_excel(writer, sheet_name='By Prop', index=False)
    
    print(f"Excel tracker created: {output_file}")
    print(f"  - {len(df)} bets loaded")
    print(f"  - Sheets: Bets, Summary, By Prop")
    print(f"\nInstructions:")
    print("  1. Open in Excel")
    print("  2. After games, fill in 'actual' column with actual stat")
    print("  3. Fill 'result' column with WIN or LOSS")
    print("  4. Profit column auto-calculates")
    print("  5. Summary sheet shows overall performance")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='betting_recommendations.csv', help='Input CSV')
    parser.add_argument('--output', default='bet_tracker.xlsx', help='Output Excel file')
    args = parser.parse_args()
    
    create_excel_tracker(args.input, args.output)


if __name__ == "__main__":
    main()
