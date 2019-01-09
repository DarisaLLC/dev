//- Common Code For All Addons needed just to ease inclusion as separate files in user code ----------------------
#include <imgui.h>
#undef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
//-----------------------------------------------------------------------------------------------------------------

#include "imguivariouscontrols.h"



namespace ImGui {

#ifndef IMGUIHELPER_H_
// Posted by Omar in one post. It might turn useful...
bool IsItemActiveLastFrame()    {
    ImGuiContext& g = *GImGui;
    if (g.ActiveIdPreviousFrame)
	return g.ActiveIdPreviousFrame== GImGui->CurrentWindow->DC.LastItemId;
    return false;
}
bool IsItemJustReleased()   {
    return IsItemActiveLastFrame() && !IsItemActive();
}
#endif //IMGUIHELPER_H_

#if 0
static float GetWindowFontScale() {
    //ImGuiContext& g = *GImGui;
    ImGuiWindow* window = GetCurrentWindow();
    return window->FontWindowScale;
}

#endif

// Start PlotHistogram(...) implementation -------------------------------
struct ImGuiPlotMultiArrayGetterData    {
    const float** Values;int Stride;
    ImGuiPlotMultiArrayGetterData(const float** values, int stride) { Values = values; Stride = stride; }

    // Some static helper methods stuffed in this struct fo convenience
    inline static void GetVerticalGradientTopAndBottomColors(ImU32 c,float fillColorGradientDeltaIn0_05,ImU32& tc,ImU32& bc)  {
        if (fillColorGradientDeltaIn0_05==0) {tc=bc=c;return;}

        const bool negative = (fillColorGradientDeltaIn0_05<0);
        if (negative) fillColorGradientDeltaIn0_05=-fillColorGradientDeltaIn0_05;
        if (fillColorGradientDeltaIn0_05>0.5f) fillColorGradientDeltaIn0_05=0.5f;

        // New code:
        //#define IM_COL32(R,G,B,A)    (((ImU32)(A)<<IM_COL32_A_SHIFT) | ((ImU32)(B)<<IM_COL32_B_SHIFT) | ((ImU32)(G)<<IM_COL32_G_SHIFT) | ((ImU32)(R)<<IM_COL32_R_SHIFT))
        const int fcgi = fillColorGradientDeltaIn0_05*255.0f;
        const int R = (unsigned char) (c>>IM_COL32_R_SHIFT);    // The cast should reset upper bits (as far as I hope)
        const int G = (unsigned char) (c>>IM_COL32_G_SHIFT);
        const int B = (unsigned char) (c>>IM_COL32_B_SHIFT);
        const int A = (unsigned char) (c>>IM_COL32_A_SHIFT);

        int r = R+fcgi, g = G+fcgi, b = B+fcgi;
        if (r>255) r=255;
        if (g>255) g=255;
        if (b>255) b=255;
        if (negative) bc = IM_COL32(r,g,b,A); else tc = IM_COL32(r,g,b,A);

        r = R-fcgi; g = G-fcgi; b = B-fcgi;
        if (r<0) r=0;
        if (g<0) g=0;
        if (b<0) b=0;
        if (negative) tc = IM_COL32(r,g,b,A); else bc = IM_COL32(r,g,b,A);

        /* // Old legacy code (to remove)... [However here we lerp alpha too...]
        // Can we do it without the double conversion ImU32 -> ImVec4 -> ImU32 ?
        const ImVec4 cf = ColorConvertU32ToFloat4(c);
        ImVec4 tmp(cf.x+fillColorGradientDeltaIn0_05,cf.y+fillColorGradientDeltaIn0_05,cf.z+fillColorGradientDeltaIn0_05,cf.w+fillColorGradientDeltaIn0_05);
        if (tmp.x>1.f) tmp.x=1.f;if (tmp.y>1.f) tmp.y=1.f;if (tmp.z>1.f) tmp.z=1.f;if (tmp.w>1.f) tmp.w=1.f;
        if (negative) bc = ColorConvertFloat4ToU32(tmp); else tc = ColorConvertFloat4ToU32(tmp);
        tmp=ImVec4(cf.x-fillColorGradientDeltaIn0_05,cf.y-fillColorGradientDeltaIn0_05,cf.z-fillColorGradientDeltaIn0_05,cf.w-fillColorGradientDeltaIn0_05);
        if (tmp.x<0.f) tmp.x=0.f;if (tmp.y<0.f) tmp.y=0.f;if (tmp.z<0.f) tmp.z=0.f;if (tmp.w<0.f) tmp.w=0.f;
        if (negative) tc = ColorConvertFloat4ToU32(tmp); else bc = ColorConvertFloat4ToU32(tmp);*/
    }
    // Same as default ImSaturate, but overflowOut can be -1,0 or 1 in case of clamping:
    inline static float ImSaturate(float f,short& overflowOut)	{
        if (f < 0.0f) {overflowOut=-1;return 0.0f;}
        if (f > 1.0f) {overflowOut=1;return 1.0f;}
        overflowOut=0;return f;
    }
    // Same as IsMouseHoveringRect, but only for the X axis (we already know if the whole item has been hovered or not)
    inline static bool IsMouseBetweenXValues(float x_min, float x_max,float *pOptionalXDeltaOut=NULL, bool clip=true, bool expandForTouchInput=false)   {
        ImGuiContext& g = *GImGui;
        ImGuiWindow* window = GetCurrentWindowRead();

        if (clip) {
            if (x_min < window->ClipRect.Min.x) x_min = window->ClipRect.Min.x;
            if (x_max > window->ClipRect.Max.x) x_max = window->ClipRect.Max.x;
        }
        if (expandForTouchInput)    {x_min-=g.Style.TouchExtraPadding.x;x_max+=g.Style.TouchExtraPadding.x;}

        if (pOptionalXDeltaOut) *pOptionalXDeltaOut = g.IO.MousePos.x - x_min;
        return (g.IO.MousePos.x >= x_min && g.IO.MousePos.x < x_max);
    }
};

#if 0
static float Plot_MultiArrayGetter(void* data, int idx,int histogramIdx)    {
    ImGuiPlotMultiArrayGetterData* plot_data = (ImGuiPlotMultiArrayGetterData*)data;
    const float v = *(float*)(void*)((unsigned char*)(&plot_data->Values[histogramIdx][0]) + (size_t)idx * plot_data->Stride);
    return v;
}

int PlotHistogram(const char* label, const float** values,int num_histograms,int values_count, int values_offset, const char* overlay_text, float scale_min, float scale_max, ImVec2 graph_size, int stride,float histogramGroupSpacingInPixels,int* pOptionalHoveredHistogramIndexOut,float fillColorGradientDeltaIn0_05,const ImU32* pColorsOverride,int numColorsOverride)   {
    ImGuiPlotMultiArrayGetterData data(values, stride);
    return PlotHistogram(label, &Plot_MultiArrayGetter, (void*)&data, num_histograms, values_count, values_offset, overlay_text, scale_min, scale_max, graph_size,histogramGroupSpacingInPixels,pOptionalHoveredHistogramIndexOut,fillColorGradientDeltaIn0_05,pColorsOverride,numColorsOverride);
}
    
#endif
    
    
int PlotHistogram(const char* label, float (*values_getter)(void* data, int idx,int histogramIdx), void* data,int num_histograms, int values_count, int values_offset, const char* overlay_text, float scale_min, float scale_max, ImVec2 graph_size,float histogramGroupSpacingInPixels,int* pOptionalHoveredHistogramIndexOut,float fillColorGradientDeltaIn0_05,const ImU32* pColorsOverride,int numColorsOverride)  {
    ImGuiWindow* window = GetCurrentWindow();
    if (pOptionalHoveredHistogramIndexOut) *pOptionalHoveredHistogramIndexOut=-1;
    if (window->SkipItems) return -1;

    static const float minSingleHistogramWidth = 5.f;   // in pixels
    static const float maxSingleHistogramWidth = 100.f;   // in pixels

    ImGuiContext& g = *GImGui;
    const ImGuiStyle& style = g.Style;

    const ImVec2 label_size = CalcTextSize(label, NULL, true);
    if (graph_size.x == 0.0f) graph_size.x = CalcItemWidth();
    if (graph_size.y == 0.0f) graph_size.y = label_size.y + (style.FramePadding.y * 2);

    const ImRect frame_bb(window->DC.CursorPos, window->DC.CursorPos + ImVec2(graph_size.x, graph_size.y));
    const ImRect inner_bb(frame_bb.Min + style.FramePadding, frame_bb.Max - style.FramePadding);
    const ImRect total_bb(frame_bb.Min, frame_bb.Max + ImVec2(label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f, 0));
    ItemSize(total_bb, style.FramePadding.y);
    if (!ItemAdd(total_bb, 0)) return -1;

    // Determine scale from values if not specified
    if (scale_min == FLT_MAX || scale_max == FLT_MAX || scale_min==scale_max)   {
        float v_min = FLT_MAX;
        float v_max = -FLT_MAX;
        for (int i = 0; i < values_count; i++)  {
            for (int h=0;h<num_histograms;h++)  {
                const float v = values_getter(data, (i + values_offset) % values_count, h);
                v_min = ImMin(v_min, v);
                v_max = ImMax(v_max, v);
            }
        }
        if (scale_min == FLT_MAX  || scale_min>=scale_max) scale_min = v_min;
        if (scale_max == FLT_MAX  || scale_min>=scale_max) scale_max = v_max;
    }
    if (scale_min>scale_max) {float tmp=scale_min;scale_min=scale_max;scale_max=tmp;}

    RenderFrame(frame_bb.Min, frame_bb.Max, GetColorU32(ImGuiCol_FrameBg), true, style.FrameRounding);

    static ImU32 col_base_embedded[] = {0,IM_COL32(100,100,225,255),IM_COL32(100,225,100,255),IM_COL32(225,100,100,255),IM_COL32(150,75,225,255)};
    static const int num_colors_embedded = sizeof(col_base_embedded)/sizeof(col_base_embedded[0]);


    int v_hovered = -1, h_hovered = -1;
    if (values_count > 0 && num_histograms > 0)
    {
        const bool mustOverrideColors = (pColorsOverride && numColorsOverride>0);
        const ImU32* col_base = mustOverrideColors ? pColorsOverride : col_base_embedded;
        const int num_colors = mustOverrideColors ? numColorsOverride : num_colors_embedded;

        const int total_histograms = values_count * num_histograms;

        const bool isItemHovered = ItemHoverable(inner_bb, 0);

        if (!mustOverrideColors) col_base_embedded[0] = GetColorU32(ImGuiCol_PlotHistogram);
        const ImU32 lineCol = GetColorU32(ImGuiCol_WindowBg);
        const float lineThick = 1.f;
        const ImU32 overflowCol = lineCol;

        const ImVec2 inner_bb_extension(inner_bb.Max.x-inner_bb.Min.x,inner_bb.Max.y-inner_bb.Min.y);
        const float scale_extension = scale_max - scale_min;
        const bool hasXAxis = scale_max>=0 && scale_min<=0;
        const float xAxisSat = ImSaturate((0.0f - scale_min) / (scale_max - scale_min));
        const float xAxisSatComp = 1.f-xAxisSat;
        const float posXAxis = inner_bb.Max.y-inner_bb_extension.y*xAxisSat-0.5f;
        const bool isAlwaysNegative = scale_max<0 && scale_min<0;

        float t_step = (inner_bb.Max.x-inner_bb.Min.x-(float)(values_count-1)*histogramGroupSpacingInPixels)/(float)total_histograms;
        if (t_step<minSingleHistogramWidth) t_step = minSingleHistogramWidth;
        else if (t_step>maxSingleHistogramWidth) t_step = maxSingleHistogramWidth;

        float t1 = 0.f;
        float posY=0.f;ImVec2 pos0(0.f,0.f),pos1(0.f,0.f);
        ImU32 rectCol=0,topRectCol=0,bottomRectCol=0;float gradient = 0.f;
        short overflow = 0;bool mustSkipBorder = false;int iWithOffset=0;
        for (int i = 0; i < values_count; i++)  {
            if (i!=0) t1+=histogramGroupSpacingInPixels;
            if (t1>inner_bb.Max.x) break;
            iWithOffset = (i + values_offset) % values_count;
            for (int h=0;h<num_histograms;h++)  {
                const float v1 = values_getter(data, iWithOffset, h);

                pos0.x = inner_bb.Min.x+t1;
                pos1.x = pos0.x+t_step;

                const int col_num = h%num_colors;
                rectCol = col_base[col_num];
                if (isItemHovered &&
                //ImGui::IsMouseHoveringRect(ImVec2(pos0.x,inner_bb.Min.y),ImVec2(pos1.x,inner_bb.Max.y))
                ImGuiPlotMultiArrayGetterData::IsMouseBetweenXValues(pos0.x,pos1.x)
                ) {
                    h_hovered = h;
                    v_hovered = iWithOffset; // iWithOffset or just i ?
                    if (pOptionalHoveredHistogramIndexOut) *pOptionalHoveredHistogramIndexOut=h_hovered;
                    SetTooltip("%d: %8.4g", i, v1); // Tooltip on hover

                    if (h==0 && !mustOverrideColors) rectCol = GetColorU32(ImGuiCol_PlotHistogramHovered); // because: col_base[0] = GetColorU32(ImGuiCol_PlotHistogram);
                    else {
                        // We don't have any hover color ready, but we can calculate it based on the same logic used between ImGuiCol_PlotHistogramHovered and ImGuiCol_PlotHistogram.
                        // Note that this code is executed only once, when a histogram is hovered.
                        const ImGuiStyle& style = GetStyle();
                        ImVec4 diff = style.Colors[ImGuiCol_PlotHistogramHovered];
                        ImVec4 base = style.Colors[ImGuiCol_PlotHistogram];
                        diff.x-=base.x;diff.y-=base.y;diff.z-=base.z;diff.w-=base.w;
                        base = ColorConvertU32ToFloat4(rectCol);
                        if (style.Alpha!=0.f) base.w /= style.Alpha;	// See GetColorU32(...) for this
                        base.x+=diff.x;base.y+=diff.y;base.z+=diff.z;base.w+=diff.w;
                        base = ImVec4(ImSaturate(base.x),ImSaturate(base.y),ImSaturate(base.z),ImSaturate(base.w));
                        rectCol = GetColorU32(base);
                    }
                }
                gradient = fillColorGradientDeltaIn0_05;
                mustSkipBorder = false;

                if (!hasXAxis) {
                    if (isAlwaysNegative)   {
                        posY = inner_bb_extension.y*ImGuiPlotMultiArrayGetterData::ImSaturate((scale_max - v1) / scale_extension,overflow);
                        overflow=-overflow;

                        pos0.y = inner_bb.Min.y;
                        pos1.y = inner_bb.Min.y+posY;

                        gradient = -fillColorGradientDeltaIn0_05;
                        mustSkipBorder = (overflow==1);
                    }
                    else {
                        posY = inner_bb_extension.y*ImGuiPlotMultiArrayGetterData::ImSaturate((v1 - scale_min) / scale_extension,overflow);

                        pos0.y = inner_bb.Max.y-posY;
                        pos1.y = inner_bb.Max.y;
                        mustSkipBorder = (overflow==-1);
                    }
                }
                else if (v1>=0){
                    posY = ImGuiPlotMultiArrayGetterData::ImSaturate(v1/scale_max,overflow);
                    pos1.y = posXAxis;
                    pos0.y = posXAxis-inner_bb_extension.y*xAxisSatComp*posY;
                }
                else {
                    posY = ImGuiPlotMultiArrayGetterData::ImSaturate(v1/scale_min,overflow);
                    overflow = -overflow;	// Probably redundant
                    pos0.y = posXAxis;
                    pos1.y = posXAxis+inner_bb_extension.y*xAxisSat*posY;
                    gradient = -fillColorGradientDeltaIn0_05;
                }

                ImGuiPlotMultiArrayGetterData::GetVerticalGradientTopAndBottomColors(rectCol,gradient,topRectCol,bottomRectCol);

                window->DrawList->AddRectFilledMultiColor(pos0, pos1,topRectCol,topRectCol,bottomRectCol,bottomRectCol); // Gradient for free!

                if (overflow!=0)    {
                    // Here we draw the small arrow that indicates that the histogram is out of scale
                    const float spacing = lineThick+1;
                    const float height = inner_bb_extension.y*0.075f;
                    // Using CW order here... but I'm not sure this is correct in Dear ImGui (we should enable GL_CULL_FACE and see if it's the same output)
                    if (overflow>0)	    window->DrawList->AddTriangleFilled(ImVec2((pos0.x+pos1.x)*0.5f,inner_bb.Min.y+spacing),ImVec2(pos1.x-spacing,inner_bb.Min.y+spacing+height), ImVec2(pos0.x+spacing,inner_bb.Min.y+spacing+height),overflowCol);
                    else if (overflow<0)    window->DrawList->AddTriangleFilled(ImVec2((pos0.x+pos1.x)*0.5f,inner_bb.Max.y-spacing),ImVec2(pos0.x+spacing,inner_bb.Max.y-spacing-height), ImVec2(pos1.x-spacing,inner_bb.Max.y-spacing-height),overflowCol);
                }

                if (!mustSkipBorder) window->DrawList->AddRect(pos0, pos1,lineCol,0.f,0,lineThick);

                t1+=t_step; if (t1>inner_bb.Max.x) break;
            }
        }

        if (hasXAxis) {
            // Draw x Axis:
            const ImU32 axisCol = GetColorU32(ImGuiCol_Text);
            const float axisThick = 1.f;
            window->DrawList->AddLine(ImVec2(inner_bb.Min.x,posXAxis),ImVec2(inner_bb.Max.x,posXAxis),axisCol,axisThick);
        }
    }

    // Text overlay
    if (overlay_text)
        RenderTextClipped(ImVec2(frame_bb.Min.x, frame_bb.Min.y + style.FramePadding.y), frame_bb.Max, overlay_text, NULL, NULL, ImVec2(0.5f,0.0f));

    if (label_size.x > 0.0f)
        RenderText(ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x, inner_bb.Min.y), label);

    return v_hovered;
}
    
#if 0
int PlotHistogram2(const char* label, const float* values,int values_count, int values_offset, const char* overlay_text, float scale_min, float scale_max, ImVec2 graph_size, int stride,float fillColorGradientDeltaIn0_05,const ImU32* pColorsOverride,int numColorsOverride) {
    const float* pValues[1] = {values};
    return PlotHistogram(label,pValues,1,values_count,values_offset,overlay_text,scale_min,scale_max,graph_size,stride,0.f,NULL,fillColorGradientDeltaIn0_05,pColorsOverride,numColorsOverride);
}
#endif
    
// End PlotHistogram(...) implementation ----------------------------------
// Start PlotCurve(...) implementation ------------------------------------
int PlotCurve(const char* label, float (*values_getter)(void* data, float x,int numCurve), void* data,int num_curves,const char* overlay_text,const ImVec2 rangeY,const ImVec2 rangeX, ImVec2 graph_size,ImVec2* pOptionalHoveredValueOut,float precisionInPixels,float numGridLinesHint,const ImU32* pColorsOverride,int numColorsOverride)  {
    ImGuiWindow* window = GetCurrentWindow();
    if (pOptionalHoveredValueOut) *pOptionalHoveredValueOut=ImVec2(0,0);    // Not sure how to initialize this...
    if (window->SkipItems || !values_getter) return -1;

    if (precisionInPixels<=1.f) precisionInPixels = 1.f;

    ImGuiContext& g = *GImGui;
    const ImGuiStyle& style = g.Style;

    const ImVec2 label_size = CalcTextSize(label, NULL, true);
    if (graph_size.x == 0.0f) graph_size.x = CalcItemWidth();
    if (graph_size.y == 0.0f) graph_size.y = label_size.y + (style.FramePadding.y * 2);

    const ImRect frame_bb(window->DC.CursorPos, window->DC.CursorPos + ImVec2(graph_size.x, graph_size.y));
    const ImRect inner_bb(frame_bb.Min + style.FramePadding, frame_bb.Max - style.FramePadding);
    const ImRect total_bb(frame_bb.Min, frame_bb.Max + ImVec2(label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f, 0));
    ItemSize(total_bb, style.FramePadding.y);
    if (!ItemAdd(total_bb, 0)) return -1;
    const ImVec2 inner_bb_extension(inner_bb.Max.x-inner_bb.Min.x,inner_bb.Max.y-inner_bb.Min.y);

    ImVec2 xRange = rangeX,yRange = rangeY,rangeExtension(0,0);

    if (yRange.x>yRange.y) {float tmp=yRange.x;yRange.x=yRange.y;yRange.y=tmp;}
    else if (yRange.x==yRange.y) {yRange.x-=5.f;yRange.y+=5.f;}
    rangeExtension.y = yRange.y-yRange.x;

    if (xRange.x>xRange.y) {float tmp=xRange.x;xRange.x=xRange.y;xRange.y=tmp;}
    else if (xRange.x==xRange.y) xRange.x-=1.f;
    if (xRange.y==FLT_MAX || xRange.x==xRange.y) xRange.y = xRange.x + (rangeExtension.y/inner_bb_extension.y)*inner_bb_extension.x;
    rangeExtension.x = xRange.y-xRange.x;

    RenderFrame(frame_bb.Min, frame_bb.Max, GetColorU32(ImGuiCol_FrameBg), true, style.FrameRounding);

    static ImU32 col_base_embedded[] = {0,IM_COL32(150,150,225,255),IM_COL32(150,225,150,255),IM_COL32(225,150,150,255),IM_COL32(150,100,225,255)};
    static const int num_colors_embedded = sizeof(col_base_embedded)/sizeof(col_base_embedded[0]);


    int v_hovered=-1;int h_hovered = -1;ImVec2 hoveredValue(FLT_MAX,FLT_MAX);
    {
        const bool mustOverrideColors = (pColorsOverride && numColorsOverride>0);
        const ImU32* col_base = mustOverrideColors ? pColorsOverride : col_base_embedded;
        const int num_colors = mustOverrideColors ? numColorsOverride : num_colors_embedded;

        const bool isItemHovered = ItemHoverable(inner_bb, 0);

        if (!mustOverrideColors) col_base_embedded[0] = GetColorU32(ImGuiCol_PlotLines);

        const bool hasXAxis = yRange.y>=0 && yRange.x<=0;
        const float xAxisSat = ImSaturate((0.0f - yRange.x) / rangeExtension.y);
        const float posXAxis = inner_bb.Max.y-inner_bb_extension.y*xAxisSat;

        const bool hasYAxis = xRange.y>=0 && xRange.x<=0;
        const float yAxisSat = ImSaturate((0.0f-xRange.x) / rangeExtension.x);
        const float posYAxis = inner_bb.Min.x+inner_bb_extension.x*yAxisSat;

        // Draw additinal horizontal and vertical lines: TODO: Fix this it's wrong
        if (numGridLinesHint>=1.f)   {
            ImU32 lineCol = GetColorU32(ImGuiCol_WindowBg);
            lineCol = (((lineCol>>24)/2)<<24)|(lineCol&0x00FFFFFF);	// Halve alpha
            const float lineThick = 1.f;
            float lineSpacing = (rangeExtension.x < rangeExtension.y) ? rangeExtension.x : rangeExtension.y;
            lineSpacing/=numGridLinesHint;      // We draw 'numGridLinesHint' lines (or so) on the shortest axis
            lineSpacing = floor(lineSpacing);   // We keep integral delta
            if (lineSpacing>0)  {
                float pos=0.f;
                // Draw horizontal lines:
                float rangeCoord = floor(yRange.x);if (rangeCoord<yRange.x) rangeCoord+=lineSpacing;
                while (rangeCoord<=yRange.y) {
                    pos = inner_bb.Max.y-inner_bb_extension.y*(yRange.y-rangeCoord)/rangeExtension.y;
                    window->DrawList->AddLine(ImVec2(inner_bb.Min.x,pos),ImVec2(inner_bb.Max.x,pos),lineCol,lineThick);
                    rangeCoord+=lineSpacing;
                }
                // Draw vertical lines:
                rangeCoord = floor(xRange.x);if (rangeCoord<xRange.x) rangeCoord+=lineSpacing;
                while (rangeCoord<=xRange.y) {
                    pos = inner_bb.Max.x-inner_bb_extension.x*(xRange.y-rangeCoord)/rangeExtension.x;
                    window->DrawList->AddLine(ImVec2(pos,inner_bb.Min.y),ImVec2(pos,inner_bb.Max.y),lineCol,lineThick);
                    rangeCoord+=lineSpacing;
                }
            }
        }

        const ImU32 axisCol = GetColorU32(ImGuiCol_Text);
        const float axisThick = 1.f;
        if (hasXAxis) {
            // Draw x Axis:
            window->DrawList->AddLine(ImVec2(inner_bb.Min.x,posXAxis),ImVec2(inner_bb.Max.x,posXAxis),axisCol,axisThick);
        }
        if (hasYAxis) {
            // Draw y Axis:
            window->DrawList->AddLine(ImVec2(posYAxis,inner_bb.Min.y),ImVec2(posYAxis,inner_bb.Max.y),axisCol,axisThick);
        }

        float mouseHoverDeltaX = 0.f;float minDistanceValue=FLT_MAX;float hoveredValueXInBBCoords=-1.f;
        ImU32 curveCol=0;const float curveThick=2.f;
        ImVec2 pos0(0.f,0.f),pos1(0.f,0.f);
        float yValue=0.f,lastYValue=0.f;
        const float t_step_xScale = precisionInPixels*rangeExtension.x/inner_bb_extension.x;
        const float t1Start = xRange.x+t_step_xScale;
        const float t1Max = xRange.x + rangeExtension.x;
        for (int h=0;h<num_curves;h++)  {
            const int col_num = h%num_colors;
            curveCol = col_base[col_num];
            lastYValue = values_getter(data, xRange.x, h);
            pos0.x = inner_bb.Min.x;
            pos0.y = inner_bb.Max.y - inner_bb_extension.y*((lastYValue-yRange.x)/rangeExtension.y);

            int v_idx = 0;
            for (float t1 = t1Start; t1 < t1Max; t1+=t_step_xScale)  {
                yValue = values_getter(data, t1, h);

                pos1.x = pos0.x+precisionInPixels;
                pos1.y = inner_bb.Max.y - inner_bb_extension.y*((yValue-yRange.x)/rangeExtension.y);

                if (pos0.y>=inner_bb.Min.y && pos0.y<=inner_bb.Max.y && pos1.y>=inner_bb.Min.y && pos1.y<=inner_bb.Max.y)
                {
                    // Draw this curve segment
                    window->DrawList->AddLine(pos0, pos1,curveCol,curveThick);
                }

                if (isItemHovered && h==0 && v_hovered==-1 && ImGuiPlotMultiArrayGetterData::IsMouseBetweenXValues(pos0.x,pos1.x,&mouseHoverDeltaX)) {
                    v_hovered = v_idx;
                    hoveredValueXInBBCoords = pos0.x+mouseHoverDeltaX;
                    hoveredValue.x = xRange.x+(hoveredValueXInBBCoords - inner_bb.Min.x)*rangeExtension.x/inner_bb_extension.x;
                }

                if (v_hovered == v_idx) {
                    const float value = values_getter(data, hoveredValue.x, h);
                    float deltaYMouse = inner_bb.Min.y + (yRange.y-value)*inner_bb_extension.y/rangeExtension.y - GetIO().MousePos.y;
                    if (deltaYMouse<0) deltaYMouse=-deltaYMouse;
                    if (deltaYMouse<minDistanceValue) {
                        minDistanceValue = deltaYMouse;

                        h_hovered = h;
                        hoveredValue.y = value;
                    }
                }


                // Mandatory
                lastYValue = yValue;
                pos0 = pos1;
                ++v_idx;
            }
        }


        if (v_hovered>=0 && h_hovered>=0)   {
            if (pOptionalHoveredValueOut) *pOptionalHoveredValueOut=hoveredValue;

	    // Tooltip on hover
	    if (num_curves==1)	SetTooltip("P (%1.4f , %1.4f)", hoveredValue.x, hoveredValue.y);
	    else		SetTooltip("P%d (%1.4f , %1.4f)",h_hovered, hoveredValue.x, hoveredValue.y);

            const float circleRadius = curveThick*3.f;
            const float hoveredValueYInBBCoords = inner_bb.Min.y+(yRange.y-hoveredValue.y)*inner_bb_extension.y/rangeExtension.y;

            // We must draw a circle in (hoveredValueXInBBCoords,hoveredValueYInBBCoords)
	    if (hoveredValueYInBBCoords>=inner_bb.Min.y && hoveredValueYInBBCoords<=inner_bb.Max.y) {
                const int col_num = h_hovered%num_colors;
                curveCol = col_base[col_num];

                window->DrawList->AddCircle(ImVec2(hoveredValueXInBBCoords,hoveredValueYInBBCoords),circleRadius,curveCol,12,curveThick);
            }

        }

    }



    // Text overlay
    if (overlay_text)
        RenderTextClipped(ImVec2(frame_bb.Min.x, frame_bb.Min.y + style.FramePadding.y), frame_bb.Max, overlay_text, NULL, NULL, ImVec2(0.5f,0.0f));

    if (label_size.x > 0.0f)
        RenderText(ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x, inner_bb.Min.y), label);

    return h_hovered;
}
// End PlotCurve(...) implementation --------------------------------------

// Start PlotMultiLines(...) and PlotMultiHistograms(...)------------------------
// by @JaapSuter and @maxint (please see: https://github.com/ocornut/imgui/issues/632)
static inline ImU32 InvertColorU32(ImU32 in)
{
    ImVec4 in4 = ColorConvertU32ToFloat4(in);
    in4.x = 1.f - in4.x;
    in4.y = 1.f - in4.y;
    in4.z = 1.f - in4.z;
    return GetColorU32(in4);
}

static void PlotMultiEx(
    ImGuiPlotType plot_type,
    const char* label,
    int num_datas,
    const char** names,
    const ImColor* colors,
    float(*getter)(const void* data, int idx),
    const void * const * datas,
    int values_count,
    float scale_min,
    float scale_max,
    ImVec2 graph_size)
{
    const int values_offset = 0;

    ImGuiWindow* window = GetCurrentWindow();
    if (window->SkipItems)
        return;

    ImGuiContext& g = *GImGui;
    const ImGuiStyle& style = g.Style;

    const ImVec2 label_size = CalcTextSize(label, NULL, true);
    if (graph_size.x == 0.0f)
        graph_size.x = CalcItemWidth();
    if (graph_size.y == 0.0f)
        graph_size.y = label_size.y + (style.FramePadding.y * 2);

    const ImRect frame_bb(window->DC.CursorPos, window->DC.CursorPos + ImVec2(graph_size.x, graph_size.y));
    const ImRect inner_bb(frame_bb.Min + style.FramePadding, frame_bb.Max - style.FramePadding);
    const ImRect total_bb(frame_bb.Min, frame_bb.Max + ImVec2(label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f, 0));
    ItemSize(total_bb, style.FramePadding.y);
    if (!ItemAdd(total_bb, 0))
        return;

    // Determine scale from values if not specified
    if (scale_min == FLT_MAX || scale_max == FLT_MAX)
    {
        float v_min = FLT_MAX;
        float v_max = -FLT_MAX;
        for (int data_idx = 0; data_idx < num_datas; ++data_idx)
        {
            for (int i = 0; i < values_count; i++)
            {
                const float v = getter(datas[data_idx], i);
                v_min = ImMin(v_min, v);
                v_max = ImMax(v_max, v);
            }
        }
        if (scale_min == FLT_MAX)
            scale_min = v_min;
        if (scale_max == FLT_MAX)
            scale_max = v_max;
    }

    RenderFrame(frame_bb.Min, frame_bb.Max, GetColorU32(ImGuiCol_FrameBg), true, style.FrameRounding);

    int res_w = ImMin((int) graph_size.x, values_count) + ((plot_type == ImGuiPlotType_Lines) ? -1 : 0);
    int item_count = values_count + ((plot_type == ImGuiPlotType_Lines) ? -1 : 0);

    // Tooltip on hover
    int v_hovered = -1;
    if (ItemHoverable(inner_bb, 0))
    {
        const float t = ImClamp((g.IO.MousePos.x - inner_bb.Min.x) / (inner_bb.Max.x - inner_bb.Min.x), 0.0f, 0.9999f);
        const int v_idx = (int) (t * item_count);
        IM_ASSERT(v_idx >= 0 && v_idx < values_count);

        // std::string toolTip;
        BeginTooltip();
        const int idx0 = (v_idx + values_offset) % values_count;
        if (plot_type == ImGuiPlotType_Lines)
        {
            const int idx1 = (v_idx + 1 + values_offset) % values_count;
            Text("%8d %8d | Name", v_idx, v_idx+1);
            for (int dataIdx = 0; dataIdx < num_datas; ++dataIdx)
            {
                const float v0 = getter(datas[dataIdx], idx0);
                const float v1 = getter(datas[dataIdx], idx1);
                TextColored(colors[dataIdx], "%8.4g %8.4g | %s", v0, v1, names[dataIdx]);
            }
        }
        else if (plot_type == ImGuiPlotType_Histogram)
        {
            for (int dataIdx = 0; dataIdx < num_datas; ++dataIdx)
            {
                const float v0 = getter(datas[dataIdx], idx0);
                TextColored(colors[dataIdx], "%d: %8.4g | %s", v_idx, v0, names[dataIdx]);
            }
        }
        EndTooltip();
        v_hovered = v_idx;
    }

    for (int data_idx = 0; data_idx < num_datas; ++data_idx)
    {
        const float t_step = 1.0f / (float) res_w;

        float v0 = getter(datas[data_idx], (0 + values_offset) % values_count);
        float t0 = 0.0f;
        ImVec2 tp0 = ImVec2(t0, 1.0f - ImSaturate((v0 - scale_min) / (scale_max - scale_min)));    // Point in the normalized space of our target rectangle

        const ImU32 col_base = colors[data_idx];
        const ImU32 col_hovered = InvertColorU32(colors[data_idx]);

        //const ImU32 col_base = GetColorU32((plot_type == ImGuiPlotType_Lines) ? ImGuiCol_PlotLines : ImGuiCol_PlotHistogram);
        //const ImU32 col_hovered = GetColorU32((plot_type == ImGuiPlotType_Lines) ? ImGuiCol_PlotLinesHovered : ImGuiCol_PlotHistogramHovered);

        for (int n = 0; n < res_w; n++)
        {
            const float t1 = t0 + t_step;
            const int v1_idx = (int) (t0 * item_count + 0.5f);
            IM_ASSERT(v1_idx >= 0 && v1_idx < values_count);
            const float v1 = getter(datas[data_idx], (v1_idx + values_offset + 1) % values_count);
            const ImVec2 tp1 = ImVec2(t1, 1.0f - ImSaturate((v1 - scale_min) / (scale_max - scale_min)));

            // NB: Draw calls are merged together by the DrawList system. Still, we should render our batch are lower level to save a bit of CPU.
            ImVec2 pos0 = ImLerp(inner_bb.Min, inner_bb.Max, tp0);
            ImVec2 pos1 = ImLerp(inner_bb.Min, inner_bb.Max, (plot_type == ImGuiPlotType_Lines) ? tp1 : ImVec2(tp1.x, 1.0f));
            if (plot_type == ImGuiPlotType_Lines)
            {
                window->DrawList->AddLine(pos0, pos1, v_hovered == v1_idx ? col_hovered : col_base);
            }
            else if (plot_type == ImGuiPlotType_Histogram)
            {
                if (pos1.x >= pos0.x + 2.0f)
                    pos1.x -= 1.0f;
                window->DrawList->AddRectFilled(pos0, pos1, v_hovered == v1_idx ? col_hovered : col_base);
            }

            t0 = t1;
            tp0 = tp1;
        }
    }

    RenderText(ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x, inner_bb.Min.y), label);
}

void PlotMultiLines(
    const char* label,
    int num_datas,
    const char** names,
    const ImColor* colors,
    float(*getter)(const void* data, int idx),
    const void * const * datas,
    int values_count,
    float scale_min,
    float scale_max,
    ImVec2 graph_size)
{
    PlotMultiEx(ImGuiPlotType_Lines, label, num_datas, names, colors, getter, datas, values_count, scale_min, scale_max, graph_size);
}

void PlotMultiHistograms(
    const char* label,
    int num_hists,
    const char** names,
    const ImColor* colors,
    float(*getter)(const void* data, int idx),
    const void * const * datas,
    int values_count,
    float scale_min,
    float scale_max,
    ImVec2 graph_size)
{
    PlotMultiEx(ImGuiPlotType_Histogram, label, num_hists, names, colors, getter, datas, values_count, scale_min, scale_max, graph_size);
}
// End PlotMultiLines(...) and PlotMultiHistograms(...)--------------------------

}
