import streamlit as st
import pandas as pd
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# --- AUTH ---
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)
try:
    login_result = authenticator.login()
    if login_result is None:
        name = st.session_state.get('name')
        authentication_status = st.session_state.get('authentication_status')
        username = st.session_state.get('username')
    else:
        name, authentication_status, username = login_result
except Exception as e:
    st.error(f"Login error: {e}")
    name = None
    authentication_status = None
    username = None

if authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')
elif authentication_status:
    authenticator.logout('Logout', 'sidebar')
    st.success(f'Welcome {name}!')

    # ---- SIDEBAR FILE UPLOAD ----
    st.sidebar.title("Upload Survey File")
    uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=['xls', 'xlsx'])
    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=None)
        st.session_state['df'] = df

    # ---- TABS IN MAIN ----
    tab_labels = [
        "Cross Reference Table",
        "Single Question Analysis",
        "Raw Data Table"
    ]
    tabs = st.tabs(tab_labels)

    # --- HELPERS ---
    def extract_answers(df, group):
        col_idxs = group['col_idxs']
        options = df.iloc[1, col_idxs].tolist()
        responses = df.iloc[2:, col_idxs]
        responses.columns = options
        if len(col_idxs) == 1 or (len(col_idxs) == 2 and options[1].strip().lower() == "other (please specify)"):
            main_series = responses.iloc[:, 0]
            return main_series
        else:
            checked = (responses.notna() & (responses != ""))
            checked.columns = options
            return checked

    def get_other_free_text(df, group):
        col_idxs = group['col_idxs']
        options = df.iloc[1, col_idxs].tolist()
        responses = df.iloc[2:, col_idxs]
        responses.columns = options
        if len(col_idxs) == 2 and options[1].strip().lower() == "other (please specify)":
            mask_other = responses.iloc[:, 0] == "Other (please specify)"
            return responses.loc[mask_other, options[1]].dropna()
        return pd.Series([], dtype=str)

    def group_questions(df):
        groups = []
        curr_question = None
        curr_cols = []
        for i, q in enumerate(df.iloc[0]):
            if pd.notna(q) and str(q).strip() != '':
                if curr_cols:
                    groups.append({'question': curr_question, 'col_idxs': curr_cols})
                curr_question = q
                curr_cols = [i]
            elif curr_question is not None and (pd.isna(q) or str(q).strip() == ''):
                curr_cols.append(i)
        if curr_cols:
            groups.append({'question': curr_question, 'col_idxs': curr_cols})
        return groups

    def get_boolean_matrix(answers):
        if isinstance(answers, pd.Series):
            return pd.get_dummies(answers)
        elif isinstance(answers, pd.DataFrame):
            return answers
        else:
            raise ValueError("Unknown answer type")

    def is_ranking_block(df, group):
        col_idxs = group['col_idxs']
        responses = df.iloc[2:, col_idxs]
        numeric_cols = responses.applymap(lambda x: pd.to_numeric(x, errors="coerce")).notna().all()
        return (len(col_idxs) > 1) and numeric_cols.all()

    # ---- TAB 1: CROSS REFERENCE ----
    with tabs[0]:
        st.header("Cross Reference Table (Any Question Type)")
        if 'df' not in st.session_state:
            st.warning("Please upload a file first (see sidebar).")
        else:
            df = st.session_state['df']
            groups = group_questions(df)
            all_questions = [g['question'] for g in groups]
            q1 = st.selectbox("Select first question (rows)", all_questions, key="xref_q1")
            q2 = st.selectbox("Select second question (columns)", all_questions, key="xref_q2")
            if q1 != q2:
                group1 = next(g for g in groups if g['question'] == q1)
                group2 = next(g for g in groups if g['question'] == q2)
                answers1 = extract_answers(df, group1)
                answers2 = extract_answers(df, group2)
                matrix1 = get_boolean_matrix(answers1)
                matrix2 = get_boolean_matrix(answers2)
                options1 = matrix1.columns
                options2 = matrix2.columns

                matrix1, matrix2 = matrix1.align(matrix2, join='inner', axis=0)
                matrix = pd.DataFrame(index=options1, columns=options2, dtype=int)
                for opt1 in options1:
                    for opt2 in options2:
                        count = ((matrix1[opt1].fillna(False)) & (matrix2[opt2].fillna(False))).sum()
                        matrix.loc[opt1, opt2] = count

                graph_type = st.selectbox(
                    "Select visualization type",
                    ["Table", "Heatmap (counts)", "Heatmap (row %)", "Bar chart (row drilldown)"],
                    key="xref_graph"
                )

                st.subheader("Cross Reference Result")

                if graph_type == "Table":
                    st.dataframe(matrix)
                elif graph_type.startswith("Heatmap"):
                    try:
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        fig, ax = plt.subplots(figsize=(min(12, len(options2)), min(8, len(options1))))
                        if graph_type == "Heatmap (row %)":
                            matrix_perc = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0) * 100
                            sns.heatmap(matrix_perc.astype(float), annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, cbar=True)
                            plt.title("Row Percentage Heatmap")
                        else:
                            sns.heatmap(matrix.astype(float), annot=True, fmt=".0f", cmap="Blues", ax=ax, cbar=True)
                            plt.title("Count Heatmap")
                        plt.xlabel(q2)
                        plt.ylabel(q1)
                        st.pyplot(fig)
                    except ImportError:
                        st.info("Install matplotlib and seaborn for a heatmap plot.")
                elif graph_type == "Bar chart (row drilldown)":
                    sel_row = st.selectbox(f"Select {q1} option (row) to view as bar chart", options1, key="xref_row")
                    row_vals = matrix.loc[sel_row]
                    st.write(f"**{q1}:** {sel_row}")
                    st.dataframe(row_vals.rename("Count"))
                    st.bar_chart(row_vals)
            else:
                st.info("Please select two different questions.")

    # ---- TAB 2: SINGLE QUESTION ----
    with tabs[1]:
        st.header("Analyze a Single Question")
        if 'df' not in st.session_state:
            st.warning("Please upload a file first (see sidebar).")
        else:
            df = st.session_state['df']
            groups = group_questions(df)
            question_list = [g['question'] for g in groups]
            sel_q = st.selectbox("Select a question", question_list, key="main_q")
            group = next(g for g in groups if g['question'] == sel_q)

            if is_ranking_block(df, group):
                col_idxs = group['col_idxs']
                options = df.iloc[1, col_idxs].tolist()
                responses = df.iloc[2:, col_idxs]
                responses.columns = options
                responses = responses.apply(pd.to_numeric, errors="coerce")

                viz_type = st.selectbox(
                    "Select ranking visualization",
                    ["Table (distribution)", "Heatmap", "Stacked bar chart", "Average rank bar"],
                    key="rank_viz"
                )
                dist = pd.DataFrame(
                    {opt: responses[opt].value_counts().sort_index() for opt in options}
                ).fillna(0).astype(int)
                dist = dist.T

                if viz_type == "Table (distribution)":
                    st.write("Rows: Options, Columns: Rank (lower is more preferred)")
                    st.dataframe(dist)
                elif viz_type == "Heatmap":
                    try:
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        fig, ax = plt.subplots(figsize=(1.5 * len(dist.columns), .6 * len(dist)))
                        sns.heatmap(dist, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax, cbar=True)
                        plt.xlabel("Rank (1=Most Favoured)")
                        plt.ylabel("Option")
                        plt.title("Rank Distribution Heatmap")
                        st.pyplot(fig)
                    except ImportError:
                        st.info("Install matplotlib and seaborn for heatmap plot.")
                elif viz_type == "Stacked bar chart":
                    try:
                        import matplotlib.pyplot as plt
                        dist_plot = dist.copy()
                        fig, ax = plt.subplots(figsize=(1.5 * len(dist_plot.index), 4))
                        dist_plot.plot(kind='bar', stacked=True, ax=ax)
                        ax.set_xlabel("Option")
                        ax.set_ylabel("Number of Respondents")
                        ax.set_title("Stacked Bar Chart of Ranks by Option")
                        st.pyplot(fig)
                    except ImportError:
                        st.info("Install matplotlib for stacked bar chart.")
                elif viz_type == "Average rank bar":
                    mean_rank = responses.mean()
                    st.write("Lower is more preferred.")
                    st.dataframe(mean_rank.rename("Average Rank"))
                    st.bar_chart(mean_rank)
            else:
                answers = extract_answers(df, group)
                if isinstance(answers, pd.Series):
                    counts = answers.value_counts(dropna=True)
                    viz_type = st.selectbox("Select visualization type", ["Table", "Bar chart", "Pie chart"], key="single_q_viz")
                    if viz_type == "Table":
                        st.dataframe(counts.rename("Count"))
                    elif viz_type == "Bar chart":
                        st.bar_chart(counts)
                    elif viz_type == "Pie chart":
                        try:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots()
                            counts.plot.pie(autopct='%1.1f%%', ax=ax)
                            ax.set_ylabel("")
                            st.pyplot(fig)
                        except ImportError:
                            st.info("matplotlib not installed, pie chart unavailable.")

                    free_text = get_other_free_text(df, group)
                    if not free_text.empty:
                        with st.expander("Show free-text responses for 'Other (please specify)'"):
                            st.write(f"Total unique 'Other' responses: {free_text.nunique()}")
                            st.dataframe(free_text.rename("Other free text"))
                elif isinstance(answers, pd.DataFrame):
                    checked_counts = answers.sum()
                    checked_counts = checked_counts[checked_counts.index.notnull()]
                    viz_type = st.selectbox(
                        "Select visualization type",
                        ["Table", "Bar chart", "Pie chart"],
                        key="multi_q_viz"
                    )
                    if viz_type == "Table":
                        st.dataframe(checked_counts.rename("Checked Count"))
                    elif viz_type == "Bar chart":
                        st.bar_chart(checked_counts)
                    elif viz_type == "Pie chart":
                        try:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots()
                            checked_counts.plot.pie(autopct='%1.1f%%', ax=ax)
                            ax.set_ylabel("")
                            st.pyplot(fig)
                        except ImportError:
                            st.info("matplotlib not installed, pie chart unavailable.")

    # ---- TAB 3: RAW DATA ----
    with tabs[2]:
        st.header("Show Raw Data Table (all responses)")
        if 'df' in st.session_state:
            st.dataframe(st.session_state['df'])
        else:
            st.warning("Please upload a file first (see sidebar).")
