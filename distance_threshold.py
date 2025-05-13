import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from io import BytesIO
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Book Search Engine",
    page_icon="📚",
    layout="wide"
)

# Define local image directory
IMAGE_DIR = Path(r"C:/Users/User/Desktop/big data project/Book Covers of Amazon Data & AI Books")

# Cache the data loading to improve performance
@st.cache_data
def load_data():
    """Load the book dataset and prepare TF-IDF search"""
    try:
        # Load data from your specific CSV file
        df = pd.read_csv("books_search_ready.csv")
        
        # Create TF-IDF vectorizer and matrix
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['search_blob'])
        
        return df, vectorizer, tfidf_matrix
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def search_with_tfidf(df, vectorizer, tfidf_matrix, query, top_n=5):
    """Search books using TF-IDF and cosine similarity"""
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_indices = [idx for idx in top_indices if similarities[idx] > 0]
    results = df.iloc[top_indices].copy()
    if not results.empty:
        results['similarity'] = similarities[top_indices]
        results['similarity_pct'] = results['similarity'] * 100
    return results

def apply_filters(df, filters):
    """Apply filters to the dataframe"""
    filtered_df = df.copy()
    for column, value in filters.items():
        if column == 'price_range':
            filtered_df = filtered_df[(filtered_df['price'] >= value[0]) & 
                                      (filtered_df['price'] <= value[1])]
        elif column == 'rating_range':
            filtered_df = filtered_df[(filtered_df['avg_reviews'] >= value[0]) & 
                                      (filtered_df['avg_reviews'] <= value[1])]
        elif column == 'pages_range':
            filtered_df = filtered_df[(filtered_df['pages'] >= value[0]) & 
                                      (filtered_df['pages'] <= value[1])]
        elif column == 'languages':
            if value:
                filtered_df = filtered_df[filtered_df['language'].isin(value)]
        elif column == 'categories':
            if value:
                filtered_df = filtered_df[filtered_df['cluster_name'].isin(value)]
    return filtered_df

def display_book_details(book):
    """Display detailed information about a book"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Load book image from local directory
        try:
            image_path = IMAGE_DIR / book['matched_image']
            if image_path.is_file():
                img = Image.open(image_path)
                st.image(img, width=200)
            else:
                # Fallback placeholder
                st.image("https://via.placeholder.com/200x300?text=No+Image", width=200)
        except Exception:
            st.image("https://via.placeholder.com/200x300?text=No+Image", width=200)
        
        # Display ratings distribution
        st.subheader("Ratings Distribution")
        ratings_data = {
            '5★': book['star5'],
            '4★': book['star4'],
            '3★': book['star3'],
            '2★': book['star2'],
            '1★': book['star1']
        }
        
        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.bar(ratings_data.keys(), ratings_data.values(), color=sns.color_palette("YlOrRd", 5))
        ax.set_ylabel('Number of Reviews')
        ax.set_title(f'Review Distribution (Avg: {book["mean_rating"]:.2f}/5)')
        
        total_reviews = sum(ratings_data.values())
        if total_reviews > 0:
            for bar in bars:
                height = bar.get_height()
                percentage = height / total_reviews * 100
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
        
        st.pyplot(fig)
    
    with col2:
        st.title(book['title'])
        st.subheader(f"by {book['author']}")
        
        st.markdown(f"""
        **Price:** ${book['price']:.2f}  
        **Publisher:** {book['publisher']}  
        **Language:** {book['language']}  
        **Pages:** {int(book['pages']) if not pd.isna(book['pages']) else 'N/A'}  
        **ISBN-13:** {book['ISBN_13']}  
        **Dimensions:** {book['dimensions']}  
        **Weight:** {book['weight']} pounds  
        **Category:** {book['cluster_name']}  
        """)
        
        st.markdown(f"""
        ### Reviews
        **Average Rating:** {book['avg_reviews']:.2f}/5 ({int(book['n_reviews'])} reviews)
        """)
        
        if pd.notna(book['complete_link']):
            st.markdown(f"[View on Amazon]({book['complete_link']})")

def main():
    df, vectorizer, tfidf_matrix = load_data()
    if df is None:
        st.error("Failed to load data. Please check your data file.")
        return
    
    st.title("📚 Book Search Engine")
    st.markdown("Search for books using TF-IDF semantic search")
    
    search_col, results_col = st.columns([1, 3], gap="large")
    with search_col:
        st.header("Search and Filters")
        query = st.text_input("Search for books:", placeholder="Enter keywords (e.g., 'machine learning')")
        top_n = st.slider("Number of results:", min_value=1, max_value=20, value=5)
        st.divider()
        # Filters setup as before...
        min_price = float(df['price'].min())
        max_price = float(df['price'].max())
        price_range = st.slider("Price Range ($):", min_value=min_price, max_value=max_price, value=(min_price, max_price), step=0.01)
        rating_range = st.slider("Rating Range:", min_value=1.0, max_value=5.0, value=(1.0, 5.0), step=0.1)
        min_pages = int(df['pages'].min())
        max_pages = int(df['pages'].max())
        pages_range = st.slider("Number of Pages:", min_value=min_pages, max_value=max_pages, value=(min_pages, max_pages))
        languages = sorted(df['language'].unique().tolist())
        selected_languages = st.multiselect("Languages:", languages)
        categories = sorted(df['cluster_name'].unique().tolist())
        selected_categories = st.multiselect("Categories:", categories)
        if st.button("Reset Filters"):
            st.rerun()
    
    with results_col:
        if query:
            results = search_with_tfidf(df, vectorizer, tfidf_matrix, query, top_n)
            filters = {'price_range': price_range, 'rating_range': rating_range, 'pages_range': pages_range,
                       'languages': selected_languages, 'categories': selected_categories}
            filtered_results = apply_filters(results, filters)
            if not filtered_results.empty:
                st.success(f"Found {len(filtered_results)} books matching '{query}'")
                for _, book in filtered_results.iterrows():
                    with st.expander(f"{book['title']} by {book['author']} - {book['similarity_pct']:.1f}% match"):
                        display_book_details(book)
            else:
                st.warning(f"No books found matching '{query}' with the current filters.")
                if not results.empty:
                    st.info("Try adjusting your filters to see more results.")
        else:
            st.info("Enter a search term to find books.")
            st.subheader("Random Book Recommendations")
            sample_books = df.sample(min(3, len(df)))
            for _, book in sample_books.iterrows():
                with st.expander(f"{book['title']} by {book['author']}"):
                    display_book_details(book)

if __name__ == "__main__":
    main()
