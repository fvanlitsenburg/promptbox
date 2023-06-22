# Get results for query
'''if run_query and question:
    reset_results()
    st.session_state.question = question

    with st.spinner(
        "🧠 &nbsp;&nbsp; Performing neural search on documents... \n "
        "Do you want to optimize speed or accuracy? \n"
        "Check out the docs: https://haystack.deepset.ai/usage/optimization "
    ):
        try:
            st.session_state.results, st.session_state.raw_json = query(
                question, top_k_retriever=top_k_retriever, model = model
            )

        except JSONDecodeError as je:
            st.error("👓 &nbsp;&nbsp; An error occurred reading the results. Is the document store working?")
            return
        except Exception as e:
            logging.exception(e)
            if "The server is busy processing requests" in str(e) or "503" in str(e):
                st.error("🧑‍🌾 &nbsp;&nbsp; All our workers are busy! Try again later.")
            else:
                st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")
            return'''
if st.session_state.results:

        # Show the gold answer if we use a question of the given set
        if eval_mode and st.session_state.answer:
            st.write("## Correct answer:")
            st.write(st.session_state.answer)

        st.write("## Results:")

        for count, result in enumerate(st.session_state.results):
            if result["answer"]:
                answer, context = result["answer"], result["context"]
                start_idx = context.find(answer)
                end_idx = start_idx + len(answer)
                # Hack due to this bug: https://github.com/streamlit/streamlit/issues/3190
                st.write(
                    markdown(context[:start_idx] + str(annotation(answer, "ANSWER", "#8ef")) + context[end_idx:]),
                    unsafe_allow_html=True,
                )
                source = ""
                url, title = get_backlink(result)
                if url and title:
                    source = f"[{result['document']['meta']['title']}]({result['document']['meta']['url']})"
                else:
                    source = f"{result['source']}"
                st.markdown(f"**Relevance:** {result['relevance']} -  **Source:** {source}")

            else:
                st.info(
                    "🤔 &nbsp;&nbsp; Haystack is unsure whether any of the documents contain an answer to your question. Try to reformulate it!"
                )
                st.write("**Relevance:** ", result["relevance"])

            if eval_mode and result["answer"]:
                # Define columns for buttons
                is_correct_answer = None
                is_correct_document = None

                button_col1, button_col2, button_col3, _ = st.columns([1, 1, 1, 6])
                if button_col1.button("👍", key=f"{result['context']}{count}1", help="Correct answer"):
                    is_correct_answer = True
                    is_correct_document = True

                if button_col2.button("👎", key=f"{result['context']}{count}2", help="Wrong answer and wrong passage"):
                    is_correct_answer = False
                    is_correct_document = False

                if button_col3.button(
                    "👎👍", key=f"{result['context']}{count}3", help="Wrong answer, but correct passage"
                ):
                    is_correct_answer = False
                    is_correct_document = True

                if is_correct_answer is not None and is_correct_document is not None:
                    try:
                        send_feedback(
                            query=question,
                            answer_obj=result["_raw"],
                            is_correct_answer=is_correct_answer,
                            is_correct_document=is_correct_document,
                            document=result["document"],
                        )
                        st.success("✨ &nbsp;&nbsp; Thanks for your feedback! &nbsp;&nbsp; ✨")
                    except Exception as e:
                        logging.exception(e)
                        st.error("🐞 &nbsp;&nbsp; An error occurred while submitting your feedback!")

            st.write("___")

        if debug:
            st.subheader("REST API JSON response")
            st.write(st.session_state.raw_json)
