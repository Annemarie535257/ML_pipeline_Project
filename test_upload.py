import os
import zipfile
import tempfile
import shutil

def test_zip_structure(zip_path):
    """Test the structure of a ZIP file"""
    print(f"Testing ZIP file: {zip_path}")
    
    if not os.path.exists(zip_path):
        print(f"ZIP file not found: {zip_path}")
        return
    
    # Create temp directory
    temp_dir = os.path.join('temp_data', f'test_extract_{int(time.time())}')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print("ZIP contents:")
            for item in zip_ref.namelist():
                print(f"  - {item}")
            
            zip_ref.extractall(temp_dir)
        
        print(f"\nExtracted to: {temp_dir}")
        print("Extracted contents:")
        
        # Walk through extracted contents
        for root, dirs, files in os.walk(temp_dir):
            level = root.replace(temp_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        # Check for Disease and Healthy folders
        expected_classes = ['Disease', 'Healthy']
        for class_name in expected_classes:
            class_dir = os.path.join(temp_dir, class_name)
            if os.path.exists(class_dir):
                file_count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"\n✅ Found {class_name} directory with {file_count} images")
            else:
                print(f"\n❌ Missing {class_name} directory")
                
                # Look for nested structure
                for root, dirs, files in os.walk(temp_dir):
                    if class_name in dirs:
                        nested_path = os.path.join(root, class_name)
                        file_count = len([f for f in os.listdir(nested_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        print(f"  Found nested {class_name} at: {nested_path} with {file_count} images")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import time
    # Test with your ZIP file
    test_zip_structure("data/retrain.zip") 