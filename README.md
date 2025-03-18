# ENI Magazine Data Extraction

This application extracts structured data from ENI magazine DOCX files and compiles it into a CSV file. It processes folders of magazine articles and uses a dual-phase AI approach to extract and verify key information.

## Features

- Process folders of DOCX files organized in the ENI magazine structure
- **Dual-Phase Extraction**:
  - **Phase 1**: Extract metadata from folder structure and initial AI analysis
  - **Phase 2**: AI verification and reconciliation of metadata against article content
- Extract metadata from folder structure (Magazine, Magazine No., Author)
- Extract text and preview images from DOCX files
- **Folder Image Support**: Automatically detects and uses images in article folders
- Use OpenAI API to infer and validate fields (Abstract, Theme, Format, Geographic Area, Keywords)
- Batch processing with start/stop controls and phase selection
- Real-time CSV generation and download
- Individual file processing for both phases
- User-friendly Streamlit interface
- Set up OpenAI API key directly in the UI (no configuration files needed)
- Flexible folder structure parsing that handles exceptions and variations
- Support for both ZIP upload and local directory processing
- Comparison of original and corrected metadata

## Project Structure

```
project/
│
├── app.py                  # Main Streamlit app
├── file_manager.py         # Functions for uploading and scanning folder structure
├── docx_parser.py          # Functions to read and extract text/images from DOCX files
├── ai_extractor.py         # Functions for initial extraction (Phase 1)
├── ai_reconciler.py        # Functions for metadata reconciliation (Phase 2)
├── csv_manager.py          # Functions for aggregating data and exporting CSV
├── config.yaml             # Configuration file with OpenAI prompts, API keys, etc.
├── requirements.txt        # Dependencies
├── output/                 # Directory for CSV output
└── README.md               # This file
```

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- An OpenAI API key (can be entered directly in the application UI)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/thevisualdev/ENI-Magazine_Data_Extraction_AI.git
   cd ENI-Magazine_Data_Extraction_AI
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   
   # On Windows
   .\.venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure the application:
   ```bash
   # Create a config file from the example
   cp config.yaml.example config.yaml
   
   # Edit config.yaml to add your OpenAI API key if desired
   # Or you can enter it later through the UI
   ```

5. Run the application:
   ```bash
   python -m streamlit run app.py
   ```

## API Key Security

**IMPORTANT:** Never store your API keys in the `config.yaml` file that gets committed to version control. Follow these best practices:

1. Copy the `.env.example` file to a new file named `.env`
2. Add your OpenAI API key to the `.env` file:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```
3. The application will automatically load the API key from this `.env` file
4. The `.env` file is listed in `.gitignore` to prevent it from being committed

The application prioritizes API keys in this order:
1. Environment variables (best method)
2. User input in the Streamlit interface (temporary)
3. Config file (not recommended)

For production deployments, use proper environment variable management through your deployment platform. Never hard-code API keys in your application files.

## AI Model Configuration

The application is configured to use OpenAI's `gpt-4o-mini` model by default, which offers a good balance between cost, speed, and accuracy. You can change the model in the `config.yaml` file if needed.

The following models are fully supported:
- `gpt-4o-mini` (default)
- `gpt-4-turbo`
- `gpt-4`
- `gpt-3.5-turbo`

The application will automatically adjust the API parameters based on the model's capabilities.

## Dual-Phase Extraction Process

The application uses a two-phase approach to ensure accurate and consistent metadata:

### Phase 1: Initial Extraction
- Extracts metadata from the folder structure (magazine, magazine number, author)
- Parses document content to extract text and preview image
- Uses AI to extract initial fields (abstract, theme, format, geographic area, keywords)

### Phase 2: Metadata Reconciliation
- Provides both the extracted article text and the file path to the AI
- Verifies and corrects metadata against the actual article content
- Reconciles discrepancies between folder structure and document content
- Ensures consistent formatting and standardization across all fields
- Preserves original metadata for comparison and reference

This dual-phase approach helps address inconsistent folder structures and ensures that the final metadata accurately reflects the article content.

## Usage

1. Start the Streamlit application using one of these methods:
   
   **Method 1**: Using Python module syntax (recommended, works with virtual environments):
   ```
   python -m streamlit run app.py
   ```
   
   **Method 2**: Direct Streamlit command (requires activated virtual environment):
   ```
   # Activate virtual environment first
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   
   # Then run Streamlit
   streamlit run app.py
   ```

2. For better performance, install the Watchdog module (optional):
   ```
   # On macOS:
   xcode-select --install
   pip install watchdog
   
   # On Windows/Linux:
   pip install watchdog
   ```

3. Access the application in your browser at the provided URL (typically http://localhost:8501)

4. Enter your OpenAI API key in the sidebar (you'll see a validation message when the key is valid)

5. Choose your preferred upload method:
   - **Upload ZIP**: Upload a ZIP file containing the magazine folder structure
   - **Select Directory**: Enter the path to a local directory containing the magazine structure

6. The expected folder structure is:
   ```
   Magazine_No/
   ├── Author1/
   │   ├── Article1.docx
   │   └── Article2.docx
   └── Author2/
       └── Article3.docx
   ```

   The application can handle variations and exceptions in this structure, including:
   - Different separators between magazine and number (underscore, dash, space)
   - Different naming conventions for magazines (supports aliases)
   - Missing parts in the folder hierarchy
   - Author information in filenames

7. Process files:
   - Use the "Start Batch Processing" button to process all files
   - Or process individual files with the "Extract Data" button
   - Monitor progress in real-time
   - Download the resulting CSV at any time

8. **Tabbed Interface**: The application provides a tabbed interface that separates:
   - **Main**: The primary data view and processing controls
   - **Phase 1**: Shows files that still need initial metadata extraction
   - **Phase 2**: Shows files that have completed Phase 1 but need metadata reconciliation

9. **Batch Processing Control**: You can:
   - Start processing all files at once
   - Stop processing at any time
   - Select which phase(s) to run (Phase 1, Phase 2, or both)
   - Monitor progress with real-time counters

## Configuration

- **OpenAI API Key**: Set it directly in the application UI (the key is validated before use)
- **Advanced Settings**: Access additional settings in the "Advanced Settings" section of the sidebar
  - OpenAI model settings (model, temperature, max tokens)
  - Phase 1 Extraction Prompt: View the prompt template used for initial metadata extraction
  - Phase 2 Reconciliation Prompt: View the prompt template used for metadata verification and correction
  - Folder structure patterns

### Custom Folder Structure Configuration

You can modify the `config.yaml` file to customize the folder structure parsing:

```yaml
folder_structure:
  # Main pattern to extract Magazine and Magazine No.
  magazine_pattern: "(.+)_([0-9]+)"
  
  # Alternative patterns to handle exceptions
  alternative_patterns:
    - pattern: "(.+)-([0-9]+)"      # Handle Magazine-No format
    - pattern: "(.+)\\s+([0-9]+)"   # Handle "Magazine No" format with space
    - pattern: "([^_]+)_"           # Handle cases with just Magazine_
  
  # Known magazine names for normalization
  known_magazines:
    - name: "Orizzonti"
      aliases: ["orizzonti", "orz", "or", "orizz"]
    - name: "WE"
      aliases: ["we", "w&e", "we_eni", "we-eni"]
```

## Expected Output

The application generates a CSV file with the following columns:
- Title
- Abstract
- Preview Image
- Folder Images - lists all image files found in the article folder (JPG, PNG, etc.)
- Magazine
- Magazine No.
- Theme
- Format
- Author
- Geographic Area
- Keywords

## Advanced Prompt Engineering

The application uses advanced prompt engineering techniques to ensure consistent and standardized extraction:

- **Structured field extraction** with clear formatting guidelines
- **Case standardization** for consistent output (sentence case for abstracts, lowercase for keywords)
- **Controlled vocabulary** for themes and formats to ensure standardization
- **Standardized geographic areas** with specific naming conventions
- **Keyword extraction** to improve searchability and categorization

You can customize the prompt in the `config.yaml` file to adjust the extraction process for your specific needs.

## Troubleshooting

- **OpenAI API Issues**: Check that your API key is valid and has been entered correctly in the UI
- **API Key Not Working**: Make sure you have sufficient credits on your OpenAI account
- **File Processing Errors**: Check the console for detailed error messages
- **Missing Fields**: Use the "Retry Extraction" button to reprocess a file
- **Folder Structure Issues**: The app logs issues when it can't parse folder structures correctly. Check the console for warnings
- **Command not found: streamlit**: Use the Python module syntax `python -m streamlit run app.py` or ensure your virtual environment is activated

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## User Interface

The application features a user-friendly Streamlit interface with the following components:

### Main Interface

1. **Sidebar Controls**:
   - OpenAI API key input with validation
   - Upload method selection (ZIP or Directory)
   - Advanced settings access

2. **Main Processing Area**:
   - Tabbed interface (Main/Phase 1/Phase 2)
   - Batch processing controls
   - Progress indicators
   - File processing status
   - CSV download button

3. **File Processing**:
   - Individual file processing controls
   - Preview of extracted metadata
   - Comparison between original and corrected metadata (after Phase 2)
   - Display of article images (shows one representative image when multiple are available)

*Note: Add screenshots of your application to make the documentation more user-friendly.*

