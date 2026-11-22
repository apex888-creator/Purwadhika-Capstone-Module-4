"""Two-language UI text (English / Simplified Chinese) and the language button.

    t("key", name=value)  -> text in the current language
    tr("Car")             -> translates short labels that come from the core
                             modules (vehicle types, periods ...); unknown
                             text is returned unchanged.
"""
import streamlit as st

LANG_KEY = "lang"          # st.session_state key: "en" or "zh"

STR = {
    # ---- general / hero ----
    "page_title": ("Electronic Road Pricing", "电子道路收费系统"),
    "hero_sub": ("Upload a traffic photo or clip and get the road charge in seconds.",
                 "上传交通照片或视频，几秒钟内即可得到道路收费结果。"),
    "flow1": ("Detect vehicles", "检测车辆"),
    "flow2": ("Read plates", "识别车牌"),
    "flow3": ("Get charge", "获取收费"),

    # ---- gate ----
    "gate_title": ("Quick human check", "快速人机验证"),
    "gate_sub": ("Solve one small sum to open the ERP system.",
                 "回答一道简单算术题即可进入 ERP 系统。"),
    "gate_locked": ("Too many wrong answers.", "错误次数过多。"),
    "gate_wait": ("Please wait a few seconds and reload the page.",
                  "请等待几秒钟后重新加载页面。"),
    "gate_answer": ("Your answer", "您的答案"),
    "gate_placeholder": ("Type a number", "请输入数字"),
    "gate_continue": ("Continue", "继续"),
    "gate_wrong": ("That was not right. Try the new question.",
                   "答案不正确，请回答新的题目。"),

    # ---- model / upload ----
    "model_missing": ("Model file for **{uc}** not found. Place `{f}` in the `models/` folder.",
                      "未找到 **{uc}** 的模型文件。请将 `{f}` 放入 `models/` 文件夹。"),
    "loading_model": ("Loading ERP model...", "正在加载 ERP 模型..."),
    "model_ready": ("ERP model ready, running on {dev}", "ERP 模型已就绪，运行于 {dev}"),
    "upload_label": ("Upload Image / Video", "上传图片 / 视频"),
    "upload_help": ("Images up to {img} MB, videos up to {vid} MB.",
                    "图片最大 {img} MB，视频最大 {vid} MB。"),
    "kind_images": ("images", "图片"),
    "kind_videos": ("videos", "视频"),
    "size_err": ("File size {size:.1f} MB exceeds the {limit} MB limit for {kind}.",
                 "文件大小 {size:.1f} MB 超过{kind}的 {limit} MB 限制。"),

    # ---- section 1 ----
    "step1_title": ("Detect vehicles", "检测车辆"),
    "step1_desc": ("YOLO finds every vehicle and calculates the base charge.",
                   "YOLO 识别每一辆车并计算基础收费。"),
    "conf": ("Confidence threshold", "置信度阈值"),
    "imgsz": ("Inference size (px)", "推理尺寸（像素）"),
    "imgsz_help": ("Smaller is faster; larger can detect small objects better but "
                   "is slower. Try 480 or 640 if processing feels slow.",
                   "数值越小速度越快；数值越大越能检测小目标，但速度更慢。"
                   "如果处理太慢，可尝试 480 或 640。"),
    "frame_skip": ("Process every N frames", "每 N 帧处理一次"),
    "frame_skip_help": ("Higher is faster, but vehicle tracking is less accurate. "
                        "Use 1 or 2 when counting vehicles.",
                        "数值越大速度越快，但车辆跟踪越不准确。统计车辆时建议用 1 或 2。"),
    "max_sec": ("Maximum duration to process (seconds)", "最长处理时长（秒）"),
    "max_sec_help": ("Limits processing time and memory.", "限制处理时间和内存占用。"),
    "uploaded_img": ("Uploaded image", "已上传的图片"),
    "btn_detect": ("🔍 Detect Objects", "🔍 开始检测"),
    "fail_process": ("Failed to process file: {e}", "文件处理失败：{e}"),
    "processing_video": ("Processing video...", "正在处理视频..."),
    "detecting": ("Detecting objects...", "正在检测目标..."),
    "det_result": ("📷 Detection result", "📷 检测结果"),
    "proc_time": ("Processing time: {s:.1f} s", "处理时间：{s:.1f} 秒"),
    "det_caption": ("Detected objects (bounding boxes)", "检测到的目标（边界框）"),
    "download": ("⬇️ Download detection result", "⬇️ 下载检测结果"),
    "note_image": ("Each vehicle detected in the image is counted once.",
                   "图片中检测到的每辆车只计数一次。"),
    "h264_warn": ("H.264 encoder is not available, so the video may not play in "
                  "the browser. Use the download button, or reinstall "
                  "imageio-ffmpeg (python -m poetry sync).",
                  "H.264 编码器不可用，视频可能无法在浏览器中播放。"
                  "请使用下载按钮，或重新安装 imageio-ffmpeg（python -m poetry sync）。"),
    "frames_done": ("{n} frames processed.", "已处理 {n} 帧。"),
    "note_tracked": ("Counted as unique vehicles via tracking (must be tracked for "
                     "at least {n} frames). This is an estimate, not an exact count.",
                     "通过跟踪统计不重复的车辆（至少需被跟踪 {n} 帧）。"
                     "这是估算值，并非精确计数。"),
    "note_untracked": ("⚠️ Tracking is unavailable, so the figures shown are the "
                       "maximum count per class in a single frame.",
                       "⚠️ 跟踪功能不可用，显示的数据为单帧内每个类别的最大数量。"),
    "erp_before": ("🚦 ERP charge (before the odd-even rule)", "🚦 ERP 收费（单双号规则前）"),
    "erp_after": ("🚦 ERP charge (with the odd-even rule)", "🚦 ERP 收费（含单双号规则）"),
    "counts_title": ("📊 Object Counts", "📊 目标数量"),
    "counts_total": ("Total objects", "目标总数"),

    # ---- section 2 ----
    "step2_title": ("Read license plates", "识别车牌"),
    "step2_desc": ("fast-alpr reads each plate and applies the odd-even rule.",
                   "fast-alpr 识别每个车牌并应用单双号规则。"),
    "need_alpr": ("Plate reading needs fast-alpr. Run `python -m poetry sync` "
                  "to install the project dependencies.",
                  "车牌识别需要 fast-alpr。请运行 `python -m poetry sync` 安装项目依赖。"),
    "need_detect": ("Run the object detection above first.", "请先运行上方的目标检测。"),
    "no_tracking": ("Tracking is unavailable, so plates cannot be matched to "
                    "individual vehicles.", "跟踪功能不可用，无法将车牌与具体车辆对应。"),
    "no_vehicles": ("No vehicles were detected, so there is nothing to read.",
                    "未检测到车辆，没有可识别的内容。"),
    "ready_crops": ("{v} vehicle(s) ready, {c} crop(s) to read.",
                    "{v} 辆车已就绪，共 {c} 张裁剪图待识别。"),
    "btn_read": ("🔎 Read Plates", "🔎 识别车牌"),
    "loading_reader": ("Loading plate reader (the first run downloads the models)...",
                       "正在加载车牌识别器（首次运行会下载模型）..."),
    "reading": ("Reading plates...", "正在识别车牌..."),
    "ocr_fail": ("Plate reading failed: {e}", "车牌识别失败：{e}"),
    "ocr_result": ("OCR result", "OCR 结果"),
    "crops_title": ("🔍 Plate crops sent to OCR", "🔍 送入 OCR 的车牌裁剪图"),
    "crop_size": ("Plate crop: {w}×{h} px", "车牌裁剪图：{w}×{h} 像素"),
    "no_crop": ("(no crop)", "（无裁剪图）"),
    "not_readable": ("— not readable —", "— 无法识别 —"),
    "confidence": ("Confidence: {c:.0%}", "置信度：{c:.0%}"),

    # ---- results ----
    "no_objects": ("No objects detected. Try lowering the confidence threshold.",
                   "未检测到目标。请尝试降低置信度阈值。"),
    "total_vehicles": ("Total vehicles", "车辆总数"),
    "total_charge": ("Total charge", "总收费"),
    "delta_vs_base": ("{sign}{amt} vs base rate", "较基础费率 {sign}{amt}"),
    "rate_mult": ("Rate multiplier", "费率倍数"),
    "col_type": ("Type", "类型"),
    "col_count": ("Count", "数量"),
    "col_base": ("Base rate", "基础费率"),
    "col_final": ("Final rate", "最终费率"),
    "col_surcharged": ("Surcharged", "加收车辆"),
    "col_subtotal": ("Subtotal", "小计"),
    "col_total": ("TOTAL", "合计"),
    "chip_period": ("Period: {p} (×{m:.2f})", "时段：{p}（×{m:.2f}）"),
    "chip_density": ("Density: {lvl} (×{m:.2f}), {n} vehicles {basis}",
                     "密度：{lvl}（×{m:.2f}），{n} 辆车 {basis}"),
    "chip_total": ("Total ×{m:.2f}", "总倍数 ×{m:.2f}"),
    "capped": ("Multiplier capped from ×{raw:.2f} to ×{cap:.2f} (maximum limit).",
               "倍数已从 ×{raw:.2f} 限制为 ×{cap:.2f}（上限）。"),
    "oddeven_info": ("**Odd-even rule for {date}** ({dp} date): {hp}-numbered plates "
                     "×{sur:.2f}, {dp}-numbered plates ×1.00. Unreadable plates are "
                     "charged ×1.00 ({unread} vehicle(s)). The plate multiplier is "
                     "applied on top of the rate shown in 'Final rate'.",
                     "**{date} 的单双号规则**（{dp}日）：{hp}号车牌 ×{sur:.2f}，"
                     "{dp}号车牌 ×1.00。无法识别的车牌按 ×1.00 收费（{unread} 辆）。"
                     "车牌倍数叠加在“最终费率”之上。"),
    "other_classes": ("Classes without a rate (not counted): ", "未设费率的类别（不计入）："),
    "no_vehicles_found": ("No vehicles detected. Try lowering the confidence threshold.",
                          "未检测到车辆。请尝试降低置信度阈值。"),
    "plate_readings": ("Plate readings", "车牌识别结果"),
    "col_vehicle": ("Vehicle", "车辆"),
    "col_plate": ("Plate", "车牌"),
    "col_ocr_conf": ("OCR confidence", "OCR 置信度"),
    "col_parity": ("Parity", "单双号"),
    "col_mult": ("Multiplier", "倍数"),
    "plate_note": ("Vehicle numbers match the labels on the detection result. "
                   "Plates are masked unless 'Show full plate text' is ticked "
                   "in the sidebar.",
                   "车辆编号与检测结果上的标签一致。除非在侧边栏勾选“显示完整车牌”，"
                   "否则车牌会被遮盖。"),

    # ---- sidebar ----
    "sb_title": ("ERP rate settings", "ERP 费率设置"),
    "sb_caption": ("Changes update the charge instantly, no need to detect again.",
                   "修改后收费会立即更新，无需重新检测。"),
    "currency": ("Currency", "货币"),
    "fx": ("Exchange rate ({cur} per 1 USD), enter manually",
           "汇率（1 美元兑 {cur}），请手动输入"),
    "sb_rates": ("💵 Base rates (USD)", "💵 基础费率（美元）"),
    "rate_label": ("{name} (USD)", "{name}（美元）"),
    "sb_period": ("🕒 Time period", "🕒 时段"),
    "adj_period": ("Adjust period multipliers", "调整时段倍数"),
    "period_mult": ("{name} multiplier", "{name}倍数"),
    "cur_period": ("Current period", "当前时段"),
    "sb_density": ("🚗 Traffic density", "🚗 交通密度"),
    "raise_busy": ("Raise rates when busy", "繁忙时提高费率"),
    "busy_from": ("Busy from (vehicles)", "繁忙起点（辆）"),
    "cong_from": ("Congested from (vehicles)", "拥堵起点（辆）"),
    "busy_x": ("Busy ×", "繁忙倍数 ×"),
    "cong_x": ("Congested ×", "拥堵倍数 ×"),
    "max_mult": ("Maximum total multiplier", "总倍数上限"),
    "max_mult_help": ("Period multiplier × density multiplier will not exceed this value.",
                      "时段倍数 × 密度倍数不会超过此值。"),
    "sb_oddeven": ("🔢 Odd-even plate rule", "🔢 车牌单双号规则"),
    "sb_oddeven_cap": ("Used after you press 'Read Plates' (section 2).",
                       "点击“识别车牌”（第 2 部分）后生效。"),
    "plate_date": ("Date to apply", "适用日期"),
    "plate_date_help": ("Even date: odd-numbered plates pay the surcharge. "
                        "Odd date: even-numbered plates pay it.",
                        "双数日期：单号车牌加收费用。单数日期：双号车牌加收费用。"),
    "surcharge": ("Surcharge multiplier", "加收倍数"),
    "surcharge_help": ("Applied to vehicles whose plate number parity differs from "
                       "the date parity. Others pay ×1.0.",
                       "适用于车牌单双号与日期单双号不同的车辆，其他车辆按 ×1.0 收费。"),
    "min_conf": ("Minimum OCR confidence", "OCR 最低置信度"),
    "min_conf_help": ("Plates read below this confidence are treated as unreadable "
                      "(charged ×1.0).",
                      "低于此置信度的车牌视为无法识别（按 ×1.0 收费）。"),
    "show_full": ("Show full plate text in the table", "在表格中显示完整车牌"),
}

# short labels that come from the core / config modules
WORDS = {
    "Car": "汽车", "Van": "厢式货车", "Bus": "公交车",
    "Normal": "正常", "Peak": "高峰", "Off-peak": "非高峰", "Offpeak": "非高峰",
    "Morning peak": "早高峰", "Evening peak": "晚高峰", "Night": "夜间",
    "Light": "畅通", "Free": "畅通", "Busy": "繁忙", "Congested": "拥堵",
    "odd": "单", "even": "双", "unread": "未识别",
    "TOTAL": "合计",
}


def lang():
    return st.session_state.get(LANG_KEY, "en")


def t(key, **kw):
    en, zh = STR[key]
    s = zh if lang() == "zh" else en
    return s.format(**kw) if kw else s


def tr(text):
    """Translate a known short label; anything else is returned unchanged."""
    if lang() != "zh":
        return text
    return WORDS.get(str(text), text)


def lang_button(key):
    """A button that flips between English and Simplified Chinese."""
    def _flip():
        st.session_state[LANG_KEY] = "en" if lang() == "zh" else "zh"

    st.button("🌐 语言：简体中文" if lang() == "en" else "🌐 Language: English",
              key=key, on_click=_flip, width="stretch")


def fmt_date(d):
    """Format a date. Built by hand for Chinese, because on Windows
    strftime() cannot take non-ASCII characters in its format string."""
    if lang() == "zh":
        return f"{d.year}年{d.month}月{d.day}日"
    return d.strftime("%d %b %Y")
