from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


regex_option =  r"(-[a-zA-Z]\r?\n)*-{2}([\=a-zA-Z\-]+)"
sample_regex = r","

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    separators=[
        regex_option,
        "\n\n",
        "\n",
        " ", 
        "",
        ],
    chunk_size=256,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=True,
)
loader = PyPDFLoader("documents/grep-pages.pdf")
pages = loader.load_and_split(text_splitter)

# for idx, page in enumerate(pages):
#     print(f"------Chunk {idx} page {page.metadata['page']} ---------")
#     print(page.page_content)

def save_pdf_as_text(pdf_path, text_path):
    # Open the PDF file
    with open(pdf_path, "rb") as file:
        # Create a PDF reader object
        pdf_reader = PdfReader(file)
        # Open the text file where the content will be written
        with open(text_path, "w", encoding="utf-8") as text_file:
            # Loop through each page in the PDF
            for page in pdf_reader.pages:
                # Extract text from the page
                text = page.extract_text()
                # If text is extracted successfully, write it to the text file
                if text:
                    text_file.write(text)
                else:
                    text_file.write('No text found on this page.\n')

# # Example usage:
# pdf_path = './documents/grep-pages.pdf'  # Path to your PDF file
# text_path = './documents/grep-pages.txt'  # Path where the text file will be saved
# save_pdf_as_text(pdf_path, text_path)

# This is a long document we can split up.
with open( "./documents/grep-pages.txt",'r', encoding='utf-8') as f:
    grep_pages = f.read()


# Split the document into chunks
chunks = text_splitter.create_documents([grep_pages])

# Print the first 20 chunks
for idx, chunk in enumerate(chunks[:20]):
    print(f"------Chunk {idx}---------")
    print(chunk.page_content)

