        # =========================
        # Metriche principali
        # =========================
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Prezzo live", f"{price_live:.2f}" if price_live is not None else "n/d")
            st.caption(f"Fonte: Yahoo – {symbol}")

        with c2:
            st.metric("Numero di Graham (da sheet)", f"{gn_sheet:.2f}" if gn_sheet is not None else "n/d")
            st.caption(f"Lettera GN: {meta.get('gn_letter')}")

        with c3:
            if margin_pct is not None:
                margin_text = f"{margin_pct:.2f}%"
                if margin_pct > 33:
                    st.markdown(f"<div style='text-align:center; color:green; font-weight:bold;'>{margin_text}<br/>🟡 G</div>", unsafe_allow_html=True)
                elif margin_pct > 0:
                    st.markdown(f"<div style='text-align:center; color:green; font-weight:bold;'>{margin_text}<br/>Sottovalutata</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align:center; color:red; font-weight:bold;'>{margin_text}<br/>Sopravvalutata</div>", unsafe_allow_html=True)
            else:
                st.metric("Margine di sicurezza", "n/d")

        # =========================
        # Formula (più grande ed evidente)
        # =========================
        st.markdown("### The GN Formula")
        if gn_formula is not None:
            st.markdown(
                f"<div style='font-size:18px; font-weight:bold; color:#222;'>√(22.5 × {eps_val:.4f} × {bvps_val:.4f}) = {gn_formula:.4f}</div>",
                unsafe_allow_html=True
            )
        else:
            st.write("Formula non calcolabile (servono EPS e BVPS > 0).")
