import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # initialize model dictionary
    model = dict()
    for new_page in corpus:
        model[new_page] = 0

    # calculate probabilities
    # check if the page has any outgoing links
    num = len(corpus[page])
    if num != 0:
        # add the probability for the outgoing links
        for new_page in corpus[page]:
            model.update({new_page: DAMPING / num})

        # add the probability for all the pages
        length = len(corpus)
        for new_page in corpus:
            model.update({new_page: model[new_page] + (1-DAMPING) / length})
    else:
        # if the page has no outgoing links, then choose any page at random
        length = len(corpus)
        for new_page in corpus:
            model.update({new_page: 1/length})

    # print(model)
    return(model)


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # initialize page_rank dicitionary
    page_rank = dict()
    for page in corpus:
        page_rank[page] = 0

    # initialize list with all the pages
    pageList = list()
    for page in corpus:
        pageList.append(page)

    # randomly choose the first page
    # length = len(corpus)
    # random_num = random.randint(1, length)
    # page = str(random_num) + '.html'
    page = random.choice(list(corpus.items()))
    page = page[0]
    page_rank.update({page: page_rank[page] + 1})

    for i in range(n-1):
        # get the transition model for the current page
        model = transition_model(corpus, page, damping_factor)

        # create a list with the probabilities
        prob = list()
        for page2 in pageList:
            prob.append(model[page2])

        # get the next page
        page = random.choices(pageList, weights=prob, k=1)
        page = page[0]

        page_rank.update({page: page_rank[page] + 1})

    # calculate the page rank
    for page in page_rank:
        page_rank.update({page: page_rank[page] / n})

    # print(page_rank)
    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # initialize page_rank dictionary
    page_rank = dict()
    n = len(corpus)
    for page in corpus:
        page_rank[page] = 1 / n

    # create a dictionary where the key is a page p, and the value is a set with all the pages that link to page p
    page_link = dict()
    for page in corpus:
        page_link[page] = set()

    for page in corpus:
        for page2 in corpus[page]:
            page_link[page2].add(page)

    ok = False
    while ok == False:
        ok = True
        for page in corpus:
            new_rank = (1 - damping_factor) / n
            s = 0

            for parent_page in page_link[page]:
                # check if the parent page has any outgoing links
                num_links = len(corpus[parent_page])
                if num_links != 0:
                    s += page_rank[parent_page] / num_links
                else:
                    s += page_rank[parent_page] / n

            new_rank += damping_factor * s
            if abs(page_rank[page] - new_rank) > 0.001:
                ok = False
            page_rank[page] = new_rank

    # print(page_rank)
    return page_rank


if __name__ == "__main__":
    main()
