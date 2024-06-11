from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

regex_option =  r"(-[a-zA-Z]\r?\n)-{2}([\=a-zA-Z\-]+)"
sample_regex = r","

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    separators=[
        sample_regex,
        "\n\n",
        "\n",
        " ", 
        "",
        ],
    chunk_size=512,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=True,
)
loader = PyPDFLoader("documents/grep-pages.pdf")
pages = loader.load_and_split(text_splitter)

for idx, page in enumerate(pages):
    print(f"------Chunk {idx} page {page.metadata['page']} ---------")
    print(page.page_content)

