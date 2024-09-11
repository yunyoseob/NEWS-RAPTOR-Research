from typing import Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders import NewsURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

class RAPTOR:
    # https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb
    def __init__(self, chunk_size=1000, overlap=200):
        from app.assistant import get_chatllm_openai, get_openai_embeddings
        from app.vectordb.vectordb import get_vectorstore
        self.chunk_size=chunk_size
        self.overlap=overlap
        self.embd = get_openai_embeddings()
        self.model = get_chatllm_openai()
        self.RANDOM_SEED = 224  # Fixed seed for reproducibility
        self.vectordb = get_vectorstore("raptor_collection")
    
    async def get_documents_by_news_url(self, url: str) -> List[Document]:
        try:
            # Document Load
            urls = []
            urls.append(url)
            loader = NewsURLLoader(
                urls=urls, 
                text_mode=True,
                nlp=True,
                show_progress_bar =True
            )
            documents = await loader.aload()
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            documents = None
        return documents
    
    async def split_chunk_text(self, text: str) -> List[str]:
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlap)
            split_texts = text_splitter.split_text(text)
        except Exception as e:
            print(f"Split texts failed : {e}")
        return split_texts
    
    async def fit_transform_umap(self, n_neighbors, n_components, metric, embeddings):
        umap_result = None
        try:
            import umap
            print("import umap: fit transform called !!!")
            umap_result = umap.umap_.UMAP(
                                    n_neighbors=n_neighbors, n_components=n_components, metric=metric
                            ).fit_transform(embeddings)
        except Exception as e:
            try:
                import umap.umap_ as umap
                print("import umap.umap_ as umap : fit transform called !!!")
                umap_result = umap.UMAP(
                                    n_neighbors=n_neighbors, n_components=n_components, metric=metric
                            ).fit_transform(embeddings)
            except Exception as e:
                print(f"UMAP fit_transform error: {e}")
                return None
        return umap_result
    
    async def global_cluster_embeddings(
        self,
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
    ) -> np.ndarray:
        if n_neighbors is None:
            n_neighbors = int((len(embeddings) - 1) ** 0.5)
        umap_result = await self.fit_transform_umap(n_neighbors=n_neighbors, n_components=dim, metric=metric, embeddings=embeddings)
        return umap_result
        
    async def local_cluster_embeddings(
        self, embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
    ) -> np.ndarray:
        umap_result = await self.fit_transform_umap(n_neighbors=num_neighbors, n_components=dim, metric=metric, embeddings=embeddings)
        return umap_result
    
    async def get_optimal_clusters(
        self, embeddings: np.ndarray, max_clusters: int = 50
    ) -> int:
        random_state = self.RANDOM_SEED
        max_clusters = min(max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        return n_clusters[np.argmin(bics)]

    async def GMM_cluster(self, embeddings: np.ndarray, threshold: float, random_state: int = 0):
        n_clusters = await self.get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, n_clusters
    
    async def perform_clustering(self, embeddings: np.ndarray, dim: int, threshold: float) -> List[np.ndarray]:
        if len(embeddings) <= dim + 1:
            # Avoid clustering when there's insufficient data
            return [np.array([0]) for _ in range(len(embeddings))]

        # Global dimensionality reduction
        reduced_embeddings_global = await self.global_cluster_embeddings(embeddings, dim)
        # Global clustering
        global_clusters, n_global_clusters = await self.GMM_cluster(
            reduced_embeddings_global, threshold
        )

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        # Iterate through each global cluster to perform local clustering
        for i in tqdm(range(n_global_clusters),desc="Clustering..."):
            # Extract embeddings belonging to the current global cluster
            global_cluster_embeddings_ = embeddings[
                np.array([i in gc for gc in global_clusters])
            ]

            if len(global_cluster_embeddings_) == 0:
                continue
            if len(global_cluster_embeddings_) <= dim + 1:
                # Handle small clusters with direct assignment
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
                n_local_clusters = 1
            else:
                # Local dimensionality reduction and clustering
                reduced_embeddings_local = await self.local_cluster_embeddings(
                    global_cluster_embeddings_, dim
                )
                local_clusters, n_local_clusters = await self.GMM_cluster(
                    reduced_embeddings_local, threshold
                )

            # Assign local cluster IDs, adjusting for total clusters already processed
            for j in range(n_local_clusters):
                local_cluster_embeddings_ = global_cluster_embeddings_[
                    np.array([j in lc for lc in local_clusters])
                ]
                indices = np.where(
                    (embeddings == local_cluster_embeddings_[:, None]).all(-1)
                )[1]
                for idx in indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )

            total_clusters += n_local_clusters
        return all_local_clusters
    
    async def embed(self, texts):
        text_embeddings = self.embd.embed_documents(texts)
        text_embeddings_np = np.array(text_embeddings)
        return text_embeddings_np

    async def embed_cluster_texts(self, texts):
        text_embeddings_np = await self.embed(texts)  # Generate embeddings
        cluster_labels = await self.perform_clustering(
            text_embeddings_np, 10, 0.1
        )  # Perform clustering on the embeddings
        df = pd.DataFrame()  # Initialize a DataFrame to store the results
        df["text"] = texts  # Store original texts
        df["embd"] = list(text_embeddings_np)  # Store embeddings as a list in the DataFrame
        df["cluster"] = cluster_labels  # Store cluster labels
        return df
    
    async def fmt_txt(self, df: pd.DataFrame) -> str:
        unique_txt = df["text"].tolist()
        return "--- --- \n --- --- ".join(unique_txt)

    async def embed_cluster_summarize_texts(self, texts: List[str], level: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_clusters = await self.embed_cluster_texts(texts)

        # Prepare to expand the DataFrame for easier manipulation of clusters
        expanded_list = []

        # Expand DataFrame entries to document-cluster pairings for straightforward processing
        for index, row in df_clusters.iterrows():
            for cluster in row["cluster"]:
                expanded_list.append(
                    {"text": row["text"], "embd": row["embd"], "cluster": cluster}
                )

        # Create a new DataFrame from the expanded list
        expanded_df = pd.DataFrame(expanded_list)

        # Retrieve unique cluster identifiers for processing
        all_clusters = expanded_df["cluster"].unique()

        print(f"--Generated {len(all_clusters)} clusters--")

        # Summarization
        template = """"여기 LangChain 표현 언어 문서의 하위 집합이 있습니다.

        LangChain 표현 언어는 LangChain에서 체인을 구성하는 방법을 제공합니다.

        제공된 문서의 자세한 요약을 제공하십시오.

        문서:
        {context}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.model | StrOutputParser()

        # Format text within each cluster for summarization
        summaries = []        
        print(f"summaries : all_clusters length : {all_clusters.size}")
        for i in tqdm(all_clusters, desc="Summarizing"):
            df_cluster = expanded_df[expanded_df["cluster"] == i]
            formatted_txt = await self.fmt_txt(df_cluster)
            print(f"summary context length : {len(formatted_txt)}")
            summaries.append(await chain.ainvoke({"context": formatted_txt}))

        # Create a DataFrame to store summaries with their corresponding cluster and level
        df_summary = pd.DataFrame(
            {
                "summaries": summaries,
                "level": [level] * len(summaries),
                "cluster": list(all_clusters),
            }
        )
        return df_clusters, df_summary
    
    async def recursive_embed_cluster_summarize(
        self, texts: List[str], level: int = 1, n_levels: int = 3
    ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        print(f'--------------------level :{level-1} -> {level} --------------------')
        print(f"text list length : {len(texts)}")

        results = {}  # Dictionary to store results at each level

        # Perform embedding, clustering, and summarization for the current level
        df_clusters, df_summary = await self.embed_cluster_summarize_texts(texts, level)

        # Store the results of the current level
        results[level] = (df_clusters, df_summary)

        # Determine if further recursion is possible and meaningful
        unique_clusters = df_summary["cluster"].nunique()
        if level < n_levels and unique_clusters > 1:
            # Use summaries as the input texts for the next level of recursion
            new_texts = df_summary["summaries"].tolist()
            next_level_results = await self.recursive_embed_cluster_summarize(
                new_texts, level + 1, n_levels
            )
            # Merge the results from the next level into the current results dictionary
            results.update(next_level_results)
        return results
    
    async def insert_additional_info(self, text_list: list[str], metadata: Dict, meta_level: str) -> List[Document]:
        """
        Description: 
        text_list : news text list (leaf node)
        metadata : 
        news 요약 시 => week, day, topic, news_index, "제목", "URL"
        topic 요약 시 => week, day
        """
        insert_docs = []
        for textIdx in range(len(text_list)):
            text = text_list[textIdx]
            metadata[meta_level] = 0
            insert_docs.append(Document(page_content=text, metadata=metadata))
        return insert_docs
    
    async def insert_vectordb(self, text_lists:List, documents_list:List[Document], metadata: Dict, meta_level: str, level=1, n_levels=3):
        """
        Description: 
        text_list : news text list (leaf node)
        """
        try:
            results = await self.recursive_embed_cluster_summarize(texts=text_lists, level=level, n_levels=n_levels)
            for level in sorted(results.keys()):
                # df_cluster = results[level][0]
                df_summary = results[level][1]
                for idx, row in df_summary.iterrows():
                    text = row['summaries']
                    metadata[meta_level] = row['level']
                    metadata['cluster'] = row['cluster']
                    documents_list.append(Document(page_content=text, metadata=metadata))
            if self.vectordb is not None and documents_list is not None:
                self.vectordb.add_documents(documents_list)
            print("vectordb add documents!!!")
        except Exception as e:
            print(f"Vector DB insert failed : {e}")
        return 1

    async def load_data(self, data: pd.DataFrame, metadata: dict):
        """
        Description: 
        data : topic dataframe data
        metadata : week, day, topic
        """
        # extract url from data
        news_cnt = 0
        documents_list = []
        text_lists = []
        for idx, row in data.iterrows():
            url = row['URL']
            metadata["news_index"]= news_cnt + 1
            metadata["제목"] = row['제목']
            metadata["URL"] = url
            if news_cnt >= 10:
                print(f"Extract news text finish !!! : news count : {news_cnt}")
                break
            if isinstance(url, str) and url.strip() != "" and url is not None:
                try:
                    documents= await self.get_documents_by_news_url(url)
                    if documents is not None:
                        for document in documents:
                            # Split Text
                            text = document.page_content 
                            text_list = await self.split_chunk_text(text)
                            insert_docs = await self.insert_additional_info(text_list=text_list, metadata=metadata, meta_level="news_level")
                            # text add
                            text_lists.extend(text_list)
                            documents_list.extend(insert_docs)       
                    news_cnt += 1
                except Exception as e:
                    print(f"Failed to process URL {url}: {e}")
                    pass
        new_metadata = {}
        new_metadata["week"] = metadata["week"]
        new_metadata["day"] = metadata["day"]
        new_metadata["topic"] = metadata["topic"]
        await self.insert_vectordb(text_lists=text_lists, documents_list=documents_list, metadata=new_metadata, meta_level="news_level")