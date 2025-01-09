import os
import csv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # 进度条库

# 修改后的关键词列表
keywords = [
    "Soccer", "Football", "Basketball", "Volleyball", "Handball", "Rugby", "American Football", "Hockey",
    "Softball", "Water Polo", "Cricket", "Netball", "Baseball", "Futsal", "Gaelic Football", "Hurling",
    "Sepak Takraw", "Lacrosse", "Dodgeball", "Tennis", "Table", "Badminton", "Squash", "Racquetball",
    "Golf", "Bowling", "Bocce", "Pickleball", "Padel", "Croquet", "Beach Tennis", "Polo", "Australian Rules Football",
    "Basque Pelota", "Korfball", "Ultimate Frisbee", "Pétanque", "Quidditch", "Bossaball", "Boxing", "Kickboxing",
    "Muay Thai", "Karate", "Taekwondo", "Sanda", "Savate", "Lethwei", "Krav Maga", "Kyokushin Karate", "Wushu",
    "Capoeira", "Judo", "Brazilian Jiu-Jitsu", "Wrestling", "Sambo", "Aikido", "Catch Wrestling", "Sumo",
    "Jujutsu", "Luta Livre", "Mixed Martial Arts", "Jeet Kune Do", "Shootfighting", "Pancrase", "Kendo",
    "Eskrima", "Arnis", "Kali", "Iaido", "Kenjutsu", "Fencing", "Naginata", "Bojutsu", "Silat", "Hapkido",
    "Tai Chi", "Baguazhang", "Wing Chun", "Xingyiquan", "Salsa", "Cha-Cha-Cha", "Rumba", "Samba", "Merengue",
    "Bachata", "Cumbia", "Tango", "Mambo", "Paso Doble", "Forró", "Waltz", "Foxtrot", "Quickstep", "Jive",
    "Lindy Hop", "Charleston", "Bolero", "Hip-Hop", "Breakdance", "Popping", "Locking", "Krumping", "Dance",
    "Voguing", "Waacking", "Jazz", "Lyrical", "Contemporary", "Street Jazz", "K-Pop",
    "Ballet", "Flamenco", "Bharatanatyam", "Kathak", "Kuchipudi", "Odissi", "Kathakali", "Butoh", "Irish Stepdance",
    "Scottish Highland", "Polka", "Mazurka", "Fandango", "Céilí", "Hula", "Sirtaki", "Flamenco Sevillanas",
    "Bollywood", "Cossack", "Hora", "Two-Step", "Line Dancing", "Square", "Zouk",
    "Lambada", "Kpanlogo", "Gwara Gwara", "Azonto", "Adumu", "Umteyo", "Piano", "Electric Piano", "Organ",
    "Harpsichord", "Clavichord", "Accordion", "Melodica", "Synthesizer", "Keytar", "Celesta", "Guitar",
     "Violin", "Viola", "Cello", "Double Bass", "Harp", "Mandolin", "Banjo", "Ukulele", "Sitar",
    "Zither", "Lute", "Dobro", "Balalaika", "Erhu", "Flute", "Piccolo", "Clarinet",
    "Bass Clarinet", "Oboe", "English Horn", "Bassoon", "Contrabassoon", "Saxophone", "Recorder", "Pan Flute",
    "Shakuhachi", "Bagpipes", "Bass Flute", "Duduk", "Trumpet", "Cornet", "Flugelhorn", "French Horn", "Trombone",
    "Bass Trombone", "Tuba", "Euphonium", "Sousaphone", "Xylophone", "Glockenspiel", "Marimba", "Vibraphone",
    "Timpani", "Steelpan", "Drum Set", "Snare Drum", "Bass Drum", "Cymbals", "Hi-Hat Cymbals", "Conga", "Bongo Drums",
    "Djembe", "Tambourine", "Triangle", "Cowbell", "Castanets", "Cajón", "Tabor", "Talking Drum", "Tabla", "Bodhrán",
    "Synthesizer", "Guitar", "Bass", "Theremin", "Drum Machine", "Sampler", "Electronic Drum Kit","Archery", "Equestrian",
    "Horseback Riding", "Gymnastics", "Parkour", "Trampoline",
    "Freerunning", "Cheerleading", "Rock Climbing", "Bouldering", "Speed Climbing",
    "Ice Climbing", "Skiing", "Snowboarding", "Figure Skating", "Speed Skating", "Curling",
    "Ice Hockey", "Field Hockey", "Rollerblading", "Inline Skating", "Skateboarding",
    "Surfing", "Windsurfing", "Kitesurfing", "Wakeboarding", "Water Skiing", "Jet Skiing",
    "Canoeing", "Kayaking", "Rowing", "Dragon Boat", "Paddleboarding", "Sailing", "Yachting",
    "Diving", "Scuba Diving", "Freediving", "Snorkeling", "Swimming", "Triathlon", "Marathon",
    "Ironman", "Road Cycling", "Track Cycling", "BMX", "Mountain Biking", "Paragliding",
    "Hang Gliding", "Skydiving", "Base Jumping", "Bungee Jumping", "Cliff Diving",
    "Parachuting", "Orienteering", "Dancehall", "Twerking", "Shuffle", "Electro",
    "Belly", "Bhangra", "Kathakali", "Garba", "Kizomba", "Bachata Sensual", "Urban Kiz",
    "Pachanga", "Charleston", "Boogie-Woogie", "Robot", "Moonwalk", "Tutting",
    "Liquid", "Bone Breaking", "Daggering", "Baglama", "Sarod", "Veena", "Gayageum",
    "Koto", "Shamisen", "Morin Khuur", "Yangqin", "Dulcimer", "Saz", "Bandura", "Bouzouki",
    "Cavaquinho", "Oud", "Pipa", "Guzheng", "Tsugaru Shamisen", "Charango", "Vihuela", "Cajón",
    "Dumbek", "Surdo", "Repinique", "Timbales", "Agogo Bells", "Shekere", "Guiro", "Cabasa",
    "Udu", "Hang Drum", "Cajón Flamenco", "Ghatam", "Darbuka", "Tabla", "Systema", "Vale Tudo",
    "Shorinji Kempo", "Kalaripayattu", "Shurikenjutsu", "Ninjutsu", "Shaolin Kung Fu", "Bando",
    "Lethwei", "Bokator", "Pankration", "Savate", "Canne de Combat", "Glima", "Mongolian Wrestling",
    "Kurash", "Sambo Combat", "Hwa Rang Do", "Taekkyeon", "Shidokan", "CQC (Close Quarters Combat)",
    "Chin Na", "Krav Maga", "Eskrima", "Naginata", "Spearfighting", "Viking Fighting Styles",
    "Kenpo", "Shaolin Boxing", "Rhythmic Gymnastics", "Artistic Gymnastics", "Floor Exercise",
    "Pommel Horse", "Rings", "Vault", "Uneven Bars", "Balance Beam", "Parallel Bars", "High Bar",
    "Slacklining", "Tightrope Walking", "Juggling", "Poi Spinning", "Baton Twirling", "Fire Dancing",
    "Staff Spinning", "Diabolo", "Plate Spinning", "Flag Twirling", "Jump Rope", "Double Dutch",
    "Trapeze", "Aerial Silks", "Aerial Hoop", "Aerial Straps", "Pole Dancing", "Contortion",
    "Hand Balancing", "Acro Yoga", "Circus Arts", "Clowning", "Mime Performance"
]


# 检查文件是否包含关键词，并返回文件名和匹配的关键词
def check_keywords_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 找到所有匹配的关键词
            matched_keywords = [keyword for keyword in keywords if keyword.lower() in content.lower()]
            # 如果有匹配的关键词，返回文件名和关键词
            if matched_keywords:
                return os.path.basename(file_path), ', '.join(matched_keywords)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

# 将结果写入CSV
def write_to_csv(results, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Matched Keywords"])  # 写入CSV的表头
        for result in results:
            if result:
                writer.writerow(result)

# 主函数：并行处理所有txt文件
def main():
    input_dir = '/home/idal-01/renshan/MOTION/DATA/HumanML3D/HumanML3D/texts/'
    output_csv = '/home/idal-01/renshan/MOTION/DATA/HumanML3D/complexM.csv'

    # 获取所有txt文件路径
    all_txt_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')]

    # 使用ThreadPoolExecutor并行处理
    with ThreadPoolExecutor(max_workers=16) as executor:
        # tqdm 显示进度条
        results = list(tqdm(executor.map(check_keywords_in_file, all_txt_files), total=len(all_txt_files), desc="Processing files"))

    # 将结果写入CSV文件
    write_to_csv(results, output_csv)
    print(f"Finished processing. Results written to {output_csv}")

if __name__ == "__main__":
    main()
