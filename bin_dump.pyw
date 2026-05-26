import tkinter as tk
from optimization2 import *

BITS_COUNT = 32
U = 27  # adjustable merge threshold

_debounce_timer = None


# -------------------------------------------------------
# Bit breakdown
# -------------------------------------------------------
def int_to_bit_details(n):
    bits = bin(n)[2:].zfill(BITS_COUNT)[::-1]
    bit_list = [int(b) for b in bits]
    details = [(int(b), 2**i, i) for i, b in enumerate(bits)]
    return details, bit_list


# -------------------------------------------------------
# GUI callback
# -------------------------------------------------------
def _do_update_bits():
    bits_text.delete("1.0", tk.END)

    entry_val = entry.get()
    if not entry_val:
        return

    try:
        number = int(entry_val)
        details, bit_list = int_to_bit_details(number)
        set_powers = [2**p for b, val, p in details if b]

        # show individual set bits
        bits_text.insert(tk.END, "Set bits breakdown:\n")
        for bit, val, p in details:
            if bit:
                bits_text.insert(tk.END, f"{bit} x {val} (2^{p})\n")

        # show minimal-term representation
        if set_powers:
            for optimization in optimizations:
                terms = optimization(set_powers, U)
                bits_text.insert(
                    tk.END,
                    f"\nMinimal-term representation ({optimization.__name__}):\n",
                )
                bits_text.insert(tk.END, terms + "\n")

    except ValueError:
        bits_text.insert(tk.END, "Enter a valid integer.")


def update_bits(event=None):
    global _debounce_timer
    if _debounce_timer is not None:
        root.after_cancel(_debounce_timer)
    _debounce_timer = root.after(150, _do_update_bits)


# -------------------------------------------------------
# GUI
# -------------------------------------------------------
root = tk.Tk()
root.title("Beltmatic numbers optimizer")
root.minsize(700, 400)

root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

tk.Label(root, text="Enter integer:").grid(row=0, column=0, sticky="w", padx=5, pady=5)

entry = tk.Entry(root)
entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
entry.bind("<KeyRelease>", update_bits)
entry.focus_set()

bits_text = tk.Text(root, wrap="none", font=("Courier", 12))
bits_text.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

scrollbar = tk.Scrollbar(root, command=bits_text.yview)
bits_text.config(yscrollcommand=scrollbar.set)
scrollbar.grid(row=2, column=2, sticky="ns")

root.mainloop()
