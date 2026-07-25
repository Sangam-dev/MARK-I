import { motion } from "motion/react";
import { Settings, User, Bell, Shield, Palette, Mic, X } from "lucide-react";

interface SettingsPageProps {
  onClose: () => void;
}

export function SettingsPage({ onClose }: SettingsPageProps) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 backdrop-blur-sm"
      style={{ background: "rgba(0, 0, 0, 0.6)" }}
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-4xl max-h-[90vh] rounded-3xl p-8 backdrop-blur-xl overflow-y-auto"
        style={{
          background: "linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(31, 41, 55, 0.9) 100%)",
          border: "1px solid rgba(255, 255, 255, 0.1)",
        }}
      >
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <Settings className="w-6 h-6 text-violet-400" />
            <h2 className="text-white">Settings</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg transition-colors"
            style={{
              background: "rgba(255, 255, 255, 0.05)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
            }}
          >
            <X className="w-5 h-5 text-white/60" />
          </button>
        </div>

        <div className="space-y-6">
          <SettingsSection
            icon={User}
            title="Profile"
            description="Manage your account settings"
          >
            <SettingItem label="Name" value="John Doe" />
            <SettingItem label="Email" value="john@example.com" />
            <SettingItem label="Role" value="Administrator" />
          </SettingsSection>

          <SettingsSection
            icon={Mic}
            title="Voice Assistant"
            description="Configure voice settings"
          >
            <SettingToggle label="Wake word detection" enabled={true} />
            <SettingToggle label="Voice feedback" enabled={true} />
            <SettingItem label="Language" value="English (US)" />
          </SettingsSection>

          <SettingsSection
            icon={Bell}
            title="Notifications"
            description="Manage notification preferences"
          >
            <SettingToggle label="Desktop notifications" enabled={true} />
            <SettingToggle label="Sound alerts" enabled={false} />
            <SettingToggle label="Email digests" enabled={true} />
          </SettingsSection>

          <SettingsSection
            icon={Palette}
            title="Appearance"
            description="Customize the interface"
          >
            <SettingItem label="Theme" value="Dark" />
            <SettingItem label="Accent color" value="Violet" />
            <SettingToggle label="Animations" enabled={true} />
          </SettingsSection>

          <SettingsSection
            icon={Shield}
            title="Privacy & Security"
            description="Control your data"
          >
            <SettingToggle label="Data collection" enabled={false} />
            <SettingToggle label="Two-factor authentication" enabled={true} />
            <SettingItem label="Session timeout" value="30 minutes" />
          </SettingsSection>
        </div>
      </motion.div>
    </motion.div>
  );
}

function SettingsSection({
  icon: Icon,
  title,
  description,
  children,
}: {
  icon: any;
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="p-6 rounded-2xl"
      style={{
        background: "rgba(255, 255, 255, 0.05)",
        border: "1px solid rgba(255, 255, 255, 0.1)",
      }}
    >
      <div className="flex items-start gap-3 mb-4">
        <Icon className="w-5 h-5 text-violet-400 mt-1" />
        <div>
          <h3 className="text-white/90 mb-1">{title}</h3>
          <p className="text-sm text-white/50">{description}</p>
        </div>
      </div>
      <div className="space-y-3 ml-8">{children}</div>
    </div>
  );
}

function SettingItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-sm text-white/70">{label}</span>
      <span className="text-sm text-white/90">{value}</span>
    </div>
  );
}

function SettingToggle({ label, enabled }: { label: string; enabled: boolean }) {
  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-sm text-white/70">{label}</span>
      <button
        className="relative w-12 h-6 rounded-full transition-colors"
        style={{
          background: enabled
            ? "linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)"
            : "rgba(255, 255, 255, 0.2)",
        }}
      >
        <motion.div
          className="absolute top-1 w-4 h-4 rounded-full bg-white"
          animate={{ left: enabled ? 28 : 4 }}
          transition={{ type: "spring", stiffness: 500, damping: 30 }}
        />
      </button>
    </div>
  );
}
