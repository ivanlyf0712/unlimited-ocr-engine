#!/usr/bin/env python3
"""
CorpChat Intelligence – Streamlit App
View contacts, messages, statistics, a chat-style conversation viewer, and semantic search.
"""

import sys
import os
import json
import requests
import streamlit as st
import pandas as pd
from datetime import datetime, timezone

# ── Ensure the project root (ocr/) is on the Python path ──
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.db import get_db_connection
from core.embedding import get_embedding
from core.config import OLLAMA_URL, RAG_MODEL

# ── Page config ──
st.set_page_config(
    page_title="CorpChat Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🕵️ CorpChat Intelligence")
st.markdown("### Corporate Relationship & Chat Analytics")

# ═══════════════════════════════════════ helpers ════════════════════════════════════
@st.cache_data(ttl=30)
def fetch_contacts():
    conn = get_db_connection()
    df = pd.read_sql(
        "SELECT id, full_name, job_title, company, phone, email, userid, created_at FROM contacts ORDER BY created_at DESC",
        conn
    )
    conn.close()
    return df

@st.cache_data(ttl=30)
def fetch_messages():
    conn = get_db_connection()
    df = pd.read_sql(
        """SELECT id, msgid, open_kfid, external_userid, send_time, origin, 
                  servicer_userid, msgtype, content, label, created_at 
           FROM messages ORDER BY send_time DESC LIMIT 500""",
        conn
    )
    conn.close()
    return df

@st.cache_data(ttl=60)
def fetch_stats():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM contacts")
    total_contacts = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM messages")
    total_msgs = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT open_kfid) FROM messages")
    total_convos = cur.fetchone()[0]
    cur.execute("""
        SELECT label, COUNT(*) 
        FROM messages 
        GROUP BY label 
        ORDER BY COUNT(*) DESC
    """)
    label_counts = cur.fetchall()
    cur.close()
    conn.close()
    return total_contacts, total_msgs, total_convos, label_counts

# ── Chat viewer helpers ──
def get_contact_name_map():
    """Return a dict {userid: full_name} for all contacts."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT userid, full_name FROM contacts")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {row[0]: (row[1] if row[1] else row[0]) for row in rows}

def get_conversation_list(label_filter=None, search_term=None):
    """Return a list of distinct conversations with last message info."""
    conn = get_db_connection()
    cur = conn.cursor()

    # Build WHERE clause
    conditions = []
    params = []
    if label_filter:
        conditions.append("label = %s")
        params.append(label_filter)
    if search_term:
        conditions.append("(external_userid ILIKE %s OR servicer_userid ILIKE %s)")
        params.extend([f"%{search_term}%", f"%{search_term}%"])

    where = " AND ".join(conditions) if conditions else "TRUE"

    # Get unique conversations (one row per open_kfid)
    cur.execute(f"""
        SELECT open_kfid,
               MAX(external_userid) AS external_userid,
               MAX(servicer_userid) AS servicer_userid,
               MAX(send_time) AS last_time
        FROM messages
        WHERE {where}
        GROUP BY open_kfid
        ORDER BY last_time DESC
        LIMIT 50
    """, params)

    conversations = []
    name_map = get_contact_name_map()
    for row in cur.fetchall():
        kfid, cust, agent, last_time = row
        # Get last message content
        cur.execute("SELECT content FROM messages WHERE open_kfid = %s ORDER BY send_time DESC LIMIT 1", (kfid,))
        last_msg = cur.fetchone()
        snippet = last_msg[0][:50] + "..." if last_msg and len(last_msg[0]) > 50 else (last_msg[0] if last_msg else "")

        display_name = name_map.get(cust, cust)
        conversations.append({
            "open_kfid": kfid,
            "display_name": display_name,
            "snippet": snippet,
            "last_time": last_time,
            "cust": cust,
            "agent": agent
        })

    cur.close()
    conn.close()
    return conversations

def get_messages_for_conversation(open_kfid):
    """Fetch all messages for a given conversation, ordered by time."""
    conn = get_db_connection()
    df = pd.read_sql(
        "SELECT msgid, external_userid, servicer_userid, send_time, origin, content "
        "FROM messages WHERE open_kfid = %s ORDER BY send_time ASC",
        conn, params=(open_kfid,)
    )
    conn.close()
    return df

# ── Semantic search helpers (Tab 5) ──
def search_messages_similar(query, label_filter=None, date_from=None, date_to=None, top_k=10):
    """Semantic search over messages using pgvector cosine similarity."""
    query_vector = get_embedding(query)
    conn = get_db_connection()
    cur = conn.cursor()

    conditions = ["embedding IS NOT NULL"]
    params = []

    if label_filter:
        conditions.append("label = %s")
        params.append(label_filter)
    if date_from:
        conditions.append("send_time >= %s")
        params.append(date_from)
    if date_to:
        conditions.append("send_time <= %s")
        params.append(date_to)

    where = " AND ".join(conditions)
    sql = f"""
        SELECT m.msgid, m.content, m.send_time, m.external_userid, m.servicer_userid, m.label,
               1 - (m.embedding <=> %s::vector) AS similarity
        FROM messages m
        WHERE {where}
        ORDER BY m.embedding <=> %s::vector
        LIMIT %s
    """
    params = [query_vector] + params + [query_vector, top_k]

    cur.execute(sql, params)
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def generate_answer_from_messages(query, messages):
    """Generate a natural-language answer from top search results."""
    context_parts = []
    for m in messages[:3]:
        content = (m[1] or "")[:1000]
        if content:
            context_parts.append(content)

    if not context_parts:
        return "No relevant message content found."

    context = "\n\n---\n\n".join(context_parts)
    prompt = f"""You are a corporate intelligence analyst. Based on the message data below,
answer the user's question in one or two sentences. If you cannot answer, say "Insufficient data."

Question: {query}

Relevant messages:
{context}

Answer:"""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": RAG_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 128}
            },
            timeout=60
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        return "(Request timed out)"
    except Exception as e:
        return f"(Error: {e})"

# ═══════════════════════════════════════════════ Tabs ═══════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Contacts", "💬 Messages", "📊 Overview", "💬 Chat Viewer", "🔍 Search"
])

# ──────────── Tab 1: Contacts ────────────
with tab1:
    st.subheader("Business Card Contacts")
    df_contacts = fetch_contacts()
    if not df_contacts.empty:
        st.metric("Total Contacts", len(df_contacts))
        st.dataframe(df_contacts, use_container_width=True)
    else:
        st.info("No contacts found. Run the data generator or OCR pipeline to populate contacts.")

# ──────────── Tab 2: Messages ────────────
with tab2:
    st.subheader("WeChat Work Messages")
    df_msgs = fetch_messages()
    if not df_msgs.empty:
        st.metric("Messages (last 500)", len(df_msgs))
        origin_map = {3: "Customer", 4: "System", 5: "Agent"}
        if 'origin' in df_msgs.columns:
            df_msgs['origin'] = df_msgs['origin'].map(origin_map)
        st.dataframe(df_msgs, use_container_width=True)
    else:
        st.info("No messages found. Generate some conversations first.")

# ──────────── Tab 3: Overview ────────────
with tab3:
    st.subheader("Database Overview")
    total_contacts, total_msgs, total_convos, label_counts = fetch_stats()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Contacts", total_contacts)
    with col2:
        st.metric("Total Messages", total_msgs)
    with col3:
        st.metric("Unique Conversations", total_convos)

    st.subheader("Messages by Label")
    if label_counts:
        df_labels = pd.DataFrame(label_counts, columns=["Label", "Count"])
        st.bar_chart(df_labels.set_index("Label"))
    else:
        st.info("No labels found. Labels are assigned during message generation (e.g., normal, scam_crypto).")

# ──────────── Tab 4: Chat Viewer ────────────
with tab4:
    st.subheader("Conversation Viewer")

    # Initialize session state for selected conversation
    if "selected_kfid" not in st.session_state:
        st.session_state.selected_kfid = None

    col_left, col_right = st.columns([1, 2], gap="medium")

    with col_left:
        label_filter = st.selectbox("Label filter",
                                     options=["All", "normal_cust_service", "normal_vendor_inquiry",
                                              "scam_crypto", "scam_phishing", "scam_job"],
                                     index=0)
        search_term = st.text_input("Search participant", placeholder="name or userid")
        label_filter = None if label_filter == "All" else label_filter

        conversations = get_conversation_list(label_filter, search_term if search_term else None)

        if not conversations:
            st.info("No conversations match the filters.")
        else:
            for conv in conversations:
                kfid = conv["open_kfid"]
                name = conv["display_name"]
                snippet = conv["snippet"]
                last_time = conv["last_time"]
                # Relative time
                now = datetime.now(timezone.utc)
                if last_time.tzinfo is None:
                    # safety: assume UTC if naive (shouldn't happen with TIMESTAMPTZ)
                    last_time = last_time.replace(tzinfo=timezone.utc)
                diff = now - last_time
                if diff.days == 0:
                    time_str = "today"
                elif diff.days == 1:
                    time_str = "yesterday"
                else:
                    time_str = f"{diff.days}d ago"

                # Unique button key based on open_kfid
                if st.button(
                    f"**{name}**  \n{snippet}  \n_{time_str}_",
                    key=f"chat_{kfid}",
                    use_container_width=True
                ):
                    st.session_state.selected_kfid = kfid

    with col_right:
        if st.session_state.selected_kfid is None:
            st.info("👈 Select a conversation from the list to view the chat.")
        else:
            kfid = st.session_state.selected_kfid
            st.markdown(f"### Conversation {kfid}")
            msgs = get_messages_for_conversation(kfid)

            if msgs.empty:
                st.warning("No messages found for this conversation.")
            else:
                name_map = get_contact_name_map()
                last_date = None
                for idx, row in msgs.iterrows():
                    msg_date = row["send_time"].date()
                    if msg_date != last_date:
                        st.markdown(f"--- **{msg_date.strftime('%Y-%m-%d')}** ---")
                        last_date = msg_date

                    if row["origin"] == 3:
                        # Customer — left side, "user" role
                        sender_name = name_map.get(row["external_userid"], row["external_userid"])
                        role = "user"
                    else:
                        # Agent / system — right side, "assistant" role
                        sender_name = name_map.get(row["servicer_userid"], row.get("servicer_userid", "System"))
                        role = "assistant"

                    # WhatsApp-style: customer = left (user), agent = right (assistant)
                    with st.chat_message(role):
                        st.markdown(f"**{sender_name}**  \n{row['content']}")
                        st.caption(row["send_time"].strftime("%H:%M"))

# ──────────── Tab 5: Semantic Search ────────────
with tab5:
    st.subheader("Semantic Search over Messages")

    # Initialize session state
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "search_query" not in st.session_state:
        st.session_state.search_query = None
    if "rag_answer" not in st.session_state:
        st.session_state.rag_answer = None

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        lbl_filter = st.selectbox(
            "Label",
            options=["All", "normal_cust_service", "normal_vendor_inquiry",
                     "scam_crypto", "scam_phishing", "scam_job"],
            index=0,
            key="search_label"
        )
    with col_f2:
        date_from = st.text_input("Date from (YYYY-MM-DD)", placeholder="2024-01-01", key="search_date_from")
    with col_f3:
        date_to = st.text_input("Date to (YYYY-MM-DD)", placeholder="2024-12-31", key="search_date_to")

    col_q1, col_q2 = st.columns([4, 1])
    with col_q1:
        search_query = st.text_input("Search query", placeholder="e.g., crypto investment scam", key="search_input")
    with col_q2:
        top_k = st.slider("Results", 1, 20, 10, key="search_top_k")

    if st.button("🔍 Search", type="primary"):
        if not search_query:
            st.warning("Please enter a search query.")
        else:
            st.session_state.rag_answer = None
            lbl = None if lbl_filter == "All" else lbl_filter
            df_val = date_from if date_from else None
            dt_val = date_to if date_to else None

            with st.spinner("Searching messages..."):
                try:
                    results = search_messages_similar(search_query, lbl, df_val, dt_val, top_k)
                    st.session_state.search_results = results
                    st.session_state.search_query = search_query
                except Exception as e:
                    st.error(f"Search error: {e}")
                    st.session_state.search_results = None

    # Display results
    if st.session_state.search_results is not None:
        results = st.session_state.search_results
        if results:
            st.success(f"Found {len(results)} result(s)")

            name_map = get_contact_name_map()
            df = pd.DataFrame(results, columns=[
                "Message ID", "Content", "Send Time", "Customer ID", "Agent ID", "Label", "Similarity"
            ])
            df["Customer"] = df["Customer ID"].apply(lambda x: name_map.get(x, x))
            df["Agent"] = df["Agent ID"].apply(lambda x: name_map.get(x, x) if x else "—")
            df["Send Time"] = df["Send Time"].apply(
                lambda t: t.strftime("%Y-%m-%d %H:%M") if hasattr(t, "strftime") else str(t)
            )
            df["Similarity"] = df["Similarity"].apply(lambda x: f"{x:.4f}")
            df["Content"] = df["Content"].apply(lambda x: (x or "")[:100] + "..." if x and len(x) > 100 else (x or ""))

            st.dataframe(
                df[["Message ID", "Customer", "Agent", "Send Time", "Label", "Similarity", "Content"]],
                use_container_width=True,
                column_config={
                    "Similarity": st.column_config.TextColumn("Similarity", width="small"),
                    "Label": st.column_config.TextColumn("Label", width="small"),
                }
            )

            # Generate Answer
            st.divider()
            if st.button("🤖 Generate Answer", type="secondary"):
                with st.spinner("Generating answer..."):
                    answer = generate_answer_from_messages(
                        st.session_state.search_query, results
                    )
                    st.session_state.rag_answer = answer

            if st.session_state.rag_answer:
                st.markdown(f"**Answer:** {st.session_state.rag_answer}")
        else:
            st.info("No matching messages found.")

st.markdown("---")
st.caption("CorpChat Intelligence – powered by Unlimited‑OCR & RAG Pipeline")