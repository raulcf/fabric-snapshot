from bs4 import BeautifulSoup

from dataaccess import csv_access

if __name__ == "__main__":
    print("Process web data")

    out_path = "/Users/ra-mit/data/fabric/academic/preprocessed/"
    paths = ["/Users/ra-mit/data/fabric/academic/wiki/", "/Users/ra-mit/data/fabric/academic/webs/"]
    all_files = []
    for p in paths:
        files = csv_access.list_files_in_directory(p)
        all_files.extend(files)

    idx = 0
    for p in all_files:
        with open(p, "r") as f:
            text_lst = f.readlines()
            text = " ".join(text_lst)
            soup = BeautifulSoup(text, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            just_text = soup.get_text()

            clean_lines = (line.strip() for line in just_text.splitlines() if line)

            preprocessed = " ".join(clean_lines)

            name = (p.split("/")[-1]).split(".")[0]
            with open(out_path + name + str(idx) + ".txt", "w") as h:
                h.writelines(preprocessed)
            idx += 1
