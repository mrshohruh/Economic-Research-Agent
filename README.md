# O'zbekiston uy-joy bozori sharhi

Avtomatik tarzda OLX.uz va Uybor.uz saytlaridagi e'lonlar asosida o'zbek
tilidagi choraklik uy-joy bozori sharhini (PDF va Word) tayyorlaydi.

## O'rnatish

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
python -m playwright install chromium
```

Model ishlatish uchun kalitni `.env` fayliga qo'ying:

```bash
cp .env.example .env
# .env faylida ANTHROPIC_API_KEY yoki GROQ_API_KEY ni to'ldiring
```

## Buyruqlar

Faqat ikkita buyruq mavjud:

```bash
python run.py                 # saqlangan ma'lumotdan hisobot tayyorlaydi
python run.py --update        # yangi ma'lumot yig'adi va hisobot tayyorlaydi
```

Birinchi marta ishga tushirishda `--update` ni qo'llang: avval ma'lumot
yig'iladi va saqlanadi. Keyingi safar oddiy `python run.py` saqlangan
ma'lumotdan hisobotni qayta tayyorlaydi.

## Natija qayerda

| Papka | Mazmuni |
|---|---|
| `outputs/reports/` | PDF va Word hisobotlar |
| `outputs/olx_snapshots/` | yig'ilgan e'lonlar (JSON) |
| `outputs/tables/` | diagrammalar va CSV jadvallar |
| `outputs/runs/` | har bir ishga tushirishning jurnali |
