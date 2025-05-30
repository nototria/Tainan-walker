import pandas as pd

def main():
    input_csv = 'tainan_edges.csv'
    output_csv = 'tainan_random_routes.csv'
    sample_size = 10000

    # 讀取原始資料
    df = pd.read_csv(input_csv)

    # 隨機抽出起點和終點資料（可重複、組合自由）
    start_points = df.sample(n=sample_size, random_state=None).reset_index(drop=True)
    end_points = df.sample(n=sample_size, random_state=None).reset_index(drop=True)

    # 組合成新的 DataFrame
    df_random_routes = pd.DataFrame({
        'start_x': start_points['start_x'],
        'start_y': start_points['start_y'],
        'end_x': end_points['end_x'],
        'end_y': end_points['end_y']
    })

    # 輸出
    df_random_routes.to_csv(output_csv, index=False)
    print(f"已隨機組合 {sample_size} 組起點終點並儲存至 {output_csv}")

if __name__ == '__main__':
    main()
