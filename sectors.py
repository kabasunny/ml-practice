def get_symbols_by_sector(sector_number):
    sector_names = {
        1: "Automotive",
        2: "Technology",
        3: "Financial",
        4: "Pharmaceutical",
        5: "Food"
    }

    switcher = {
        1: get_automotive_symbols,
        2: get_technology_symbols,
        3: get_financial_symbols,
        4: get_pharmaceutical_symbols,
        5: get_food_symbols,
    }
    
    # デフォルトとしてValueErrorを設定
    func = switcher.get(sector_number, lambda: ValueError("Invalid sector number"))
    symbols = func()
    
    # セクター名を取得
    sector_name = sector_names.get(sector_number, "Unknown")
    
    # 選択されたセクターの銘柄リストを表示
    print(f"Selected sector ({sector_number} - {sector_name})")
    print(f"symbols: {symbols}")
    
    return symbols

def get_automotive_symbols():
    return [
        "7203.T",  # Toyota Motor Corporation
        "7201.T",  # Nissan Motor Co., Ltd.
        "7267.T",  # Honda Motor Co., Ltd.
        "7261.T",  # Mazda Motor Corporation
        "7269.T",  # Suzuki Motor Corporation
        # "7262.T",  # Mitsubishi Motors Corporation 上場廃止
        "7270.T",  # Subaru Corporation
        "7202.T",  # Isuzu Motors Limited
        "7205.T",  # Hino Motors, Ltd.
        "7211.T",  # Mitsubishi Fuso Truck and Bus Corporation
        "7224.T",  # Shizuoka Daihatsu Motor Co., Ltd.
        "7266.T"   # Showa Corporation
    ]

def get_technology_symbols():
    return [
        "6758.T",  # Sony Corporation
        "6702.T",  # Fujitsu Limited
        "6971.T",  # Kyocera Corporation
        "6501.T",  # Hitachi, Ltd.
        "7751.T",  # Canon Inc.
        "6752.T",  # Panasonic Corporation
        "7974.T",  # Nintendo Co., Ltd.
        "6701.T",  # NEC Corporation
        "6703.T",  # Oki Electric Industry Co., Ltd.
        "6724.T",  # Seiko Epson Corporation
        "9684.T",  # Square Enix Holdings Co., Ltd.
        "9766.T",  # Konami Holdings Corporation
        "9697.T"   # Capcom Co., Ltd.
    ]

def get_financial_symbols():
    return [
        "8306.T",  # Mitsubishi UFJ Financial Group, Inc.
        # "8316.T",  # Sumitomo Mitsui Financial Group, Inc.
        # "8411.T",  # Mizuho Financial Group, Inc.
        # "8604.T",  # Nomura Holdings, Inc.
        # "8628.T"   # Matsui Securities Co., Ltd.
    ]

def get_pharmaceutical_symbols():
    return [
        "4502.T",  # Takeda Pharmaceutical Company Limited
        "4503.T",  # Astellas Pharma Inc.
        "4506.T",  # Daiichi Sankyo Company, Limited
        "4519.T",  # Chugai Pharmaceutical Co., Ltd.
        "4523.T"   # Eisai Co., Ltd.
    ]

def get_food_symbols():
    return [
        "2502.T",  # Asahi Group Holdings, Ltd.
        "2503.T",  # Kirin Holdings Company, Limited
        "2802.T",  # Ajinomoto Co., Inc.
        "2801.T",  # Kikkoman Corporation
        "2914.T"   # Japan Tobacco Inc.
    ]
