def encode_feature(value, mapping):
    return mapping.get(value, -1)

def preprocess_input(last_laid_year, last_repair_year, material, weather, usage_type, traffic_level, accidents_reported):
    road_age = 2025 - last_laid_year
    years_since_repair = 2025 - last_repair_year

    material_map = {'asphalt': 0, 'concrete': 1, 'gravel': 2}
    weather_map = {'hot': 0, 'humid': 1, 'rainy': 2}
    usage_map = {'urban': 0, 'rural': 1, 'highway': 2}
    traffic_map = {'low': 0, 'medium': 1, 'high': 2}

    return [
        road_age,
        years_since_repair,
        encode_feature(material, material_map),
        encode_feature(weather, weather_map),
        encode_feature(usage_type, usage_map),
        encode_feature(traffic_level, traffic_map),
        accidents_reported
    ]
