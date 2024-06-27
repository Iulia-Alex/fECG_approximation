import os
import shutil

def move_files(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        if os.path.isfile(source_path):
            dest_path = os.path.join(destination_dir, filename)
            base, extension = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(destination_dir, f"{base}_{counter}{extension}")
                counter += 1
            shutil.move(source_path, dest_path)
            print(f"Moved {source_path} to {dest_path}")

def rename_files(destination_dir):
    files = sorted([f for f in os.listdir(destination_dir) if f.startswith("fecgsyn") and f.endswith(".mat")])
    index = 0
    for filename in files:
        expected_name = f"fecgsyn{index}.mat"
        source_path = os.path.join(destination_dir, filename)
        dest_path = os.path.join(destination_dir, expected_name)

        if filename != expected_name:
            base, extension = os.path.splitext(expected_name)
            while os.path.exists(dest_path):
                index += 1
                expected_name = f"fecgsyn{index}.mat"
                dest_path = os.path.join(destination_dir, expected_name)
            shutil.move(source_path, dest_path)
            print(f"Renamed {source_path} to {dest_path}")
        index += 1

if __name__ == "__main__":
    source_dir = "C:/Users/ovidiu/Desktop/ecg"
    destination_dir = "C:/Users/ovidiu/Download/ecg"
    
    move_files(source_dir, destination_dir)
    rename_files(destination_dir)

    print("Finished processing files.")