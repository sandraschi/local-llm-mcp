import {
  Activity,
  Bot,
  Brain,
  ChevronLeft,
  ChevronRight,
  Gauge,
  Image as ImageIcon,
  LayoutDashboard,
  LayoutGrid,
  Settings,
} from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { cn } from "@/common/utils";

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const location = useLocation();

  const navItems = [
    { href: "/", label: "Overview", icon: LayoutDashboard },
    { href: "/chat", label: "Chat", icon: Bot },
    { href: "/performance", label: "Performance", icon: Gauge },
    { href: "/vision", label: "Vision", icon: ImageIcon },
    { href: "/fleet", label: "Fleet", icon: LayoutGrid },
    { href: "/analytics", label: "Analytics", icon: Activity },
    { href: "/settings", label: "Settings", icon: Settings },
  ];

  return (
    <aside
      className={cn(
        "glass-sidebar relative flex flex-col transition-all duration-300 ease-in-out",
        collapsed ? "w-20" : "w-64",
      )}
    >
      <div className="flex h-20 items-center px-6">
        <div className="flex items-center gap-3 font-bold text-slate-100">
          <div className="w-8 h-8 rounded-lg bg-emerald-600 flex items-center justify-center emerald-glow">
            <Brain className="h-5 w-5 text-white" />
          </div>
          {!collapsed && <span className="text-lg font-bold gradient-text">LOCAL-LLM</span>}
        </div>
      </div>

      <nav className="flex-1 space-y-1 p-3">
        {navItems.map((item) => {
          const isActive = location.pathname === item.href;
          return (
            <Link
              key={item.href}
              to={item.href}
              className={cn(
                "nav-item",
                isActive && "active",
                collapsed ? "justify-center" : "justify-start",
              )}
            >
              <item.icon className={cn("h-5 w-5", isActive && "text-emerald-400")} />
              {!collapsed && <span>{item.label}</span>}

              {collapsed && (
                <div className="absolute left-full ml-4 hidden rounded-lg bg-black/80 backdrop-blur-md px-3 py-1.5 text-xs text-white group-hover:block z-50 whitespace-nowrap border border-white/10">
                  {item.label}
                </div>
              )}
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-white/[0.06]">
        <button
          type="button"
          onClick={onToggle}
          className="flex w-full items-center justify-center rounded-xl p-2.5 text-slate-400 hover:text-white hover:bg-white/[0.05] transition-all"
        >
          {collapsed ? (
            <ChevronRight className="h-5 w-5" />
          ) : (
            <div className="flex items-center w-full">
              <ChevronLeft className="h-5 w-5 mr-3" />
              <span>Collapse Sidebar</span>
            </div>
          )}
        </button>
      </div>
    </aside>
  );
}
